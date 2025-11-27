import asyncio
import json
import uuid
import os
from chatchat.server.db.repository.message_repository import filter_message
from typing import AsyncIterable, List, Union, Tuple
from langchain_core.load import dumpd, dumps, load, loads

from fastapi import Body
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate

from chatchat.server.agents_registry.agents_registry import agents_registry
from sse_starlette.sse import EventSourceResponse

from chatchat.server.db.repository.mcp_connection_repository import get_enabled_mcp_connections
from chatchat.settings import Settings
from chatchat.server.api_server.api_schemas import OpenAIChatOutput
from langchain_chatchat.callbacks.agent_callback_handler import (
    AgentExecutorAsyncIteratorCallbackHandler,
    AgentStatus,
)
from langchain_chatchat.agents.platform_tools import PlatformToolsAction, PlatformToolsFinish, \
    PlatformToolsActionToolStart, PlatformToolsActionToolEnd, PlatformToolsLLMStatus
from chatchat.server.chat.utils import History
from chatchat.server.db.repository import add_message_to_db, update_message

from langchain_chatchat import ChatPlatformAI, PlatformToolsRunnable
from chatchat.server.utils import (
    MsgType,
    get_ChatOpenAI,
    get_prompt_template,
    get_tool,
    wrap_done,
    get_default_llm,
    build_logger,
    get_ChatPlatformAIParams
)

logger = build_logger()


def create_models_from_config(configs, callbacks, stream, max_tokens):
    # 、、根据传入的模型配置生成模型实例，（优先用外部传入的模型配置，没有穿就用配置文件里的，）
    configs = configs or Settings.model_settings.LLM_MODEL_CONFIG
    models = {}
    prompts = {}
    for model_type, params in configs.items():
        # 、、遍历每一个配置，获取模型 的名称，这里配置文件里也是空字符传，所以会去 or get_default_llm 中获取(会从另外一个配置文件中去获取)
        model_name = params.get("model", "").strip() or get_default_llm()
        # 没有callbacks就直接赋值为None，如果前面的监控有加就会有callback
        callbacks = callbacks if params.get("callbacks", False) else None
        # 判断是否传入 max_tokens 的值, 如果传入就按传入的赋值(api 调用且赋值), 如果没有传入则按照初始化配置赋值(ui 调用或 api 调用未赋值)
        max_tokens_value = max_tokens if max_tokens is not None else params.get("max_tokens", 1000)
        # 、、如果当 模型是 行动模型
        if model_type == "action_model":
            # 、、
            llm_params = get_ChatPlatformAIParams(
                model_name=model_name,
                temperature=params.get("temperature", 0.5),
                max_tokens=max_tokens_value,
            )
            model_instance = ChatPlatformAI(**llm_params)
        else:
            model_instance = get_ChatOpenAI(
                model_name=model_name,
                temperature=params.get("temperature", 0.5),
                max_tokens=max_tokens_value,
                callbacks=callbacks,
                streaming=stream,
                local_wrap=True,
            )
        models[model_type] = model_instance
        prompt_name = params.get("prompt_name", "default")
        # 、、根据 入参中的， model_type和提示词名称 （没传就默认用default）获取提示词，
        prompt_template = get_prompt_template(type=model_type, name=prompt_name)
        prompts[model_type] = prompt_template
    #、、 返回models(是个字典，model_type字段上来了模型实例)，返回提示词 
    return models, prompts


def   create_models_chains(
    history_len, prompts, models, tools, callbacks, conversation_id, metadata,  use_mcp: bool = False
):

    # 从数据库获取conversation_id对应的 intermediate_steps 、 mcp_connections
    messages = filter_message(
        conversation_id=conversation_id, limit=history_len
    )
    # 返回的记录按时间倒序，转为正序（数据库中来的数据,最新的用户输入也已经入库了）
    messages = list(reversed(messages))
    history: List[Union[List, Tuple]] = []
    for message in messages:
        # 、、构造消息记录
        history.append({"role": "user", "content": message["query"]}) 
        history.append({"role": "assistant", "content":  message["response"]})  


    intermediate_steps = loads(
        messages[-1].get("metadata", {}).get("intermediate_steps"), 
        valid_namespaces=["langchain_chatchat", "agent_toolkits", "all_tools", "tool"] 
        )  if len(messages)>0 and messages[-1].get("metadata") is not None else []
    """、、从最新的元数据中回复Agent的执行状态
    message[-1] 获取最新的一条消息
    metadata.intermeditate_steps 存储Agent执行过程中的工具调用记录
    loads() 反序列化存储的中间步骤数据
    valid_name_spaces 指定允许的反序列化命名空间，确保安全
        中间步骤数据结构
            intermediate_step = [
                (tool_call, tool_output), #第一次工具调用和结果
                (tool_call, tool_output), #第二次工具调用和结果
            ]
    """
    # 、、获取行动模型
    llm = models["action_model"]
    # 、、给模型设置调函数
    llm.callbacks = callbacks
    # 、、数据库查询所有启用的mcp连接器
    connections = get_enabled_mcp_connections()
    
    # 转换为MCP连接格式，支持StdioConnection和SSEConnection类型
    mcp_connections = {}
    for conn in connections:
        if conn["transport"] == "stdio":
            # StdioConnection类型
            mcp_connections[conn["server_name"]] = {
                "transport": "stdio",
                "command": conn["config"].get("command", conn["args"][0] if conn["args"] else ""),
                "args": conn["args"][1:] if len(conn["args"]) > 1 else [],
                "env": conn["env"],
                "encoding": "utf-8",
                "encoding_error_handler": "strict"
            }
        elif conn["transport"] == "sse":
            # SSEConnection类型
            mcp_connections[conn["server_name"]] = {
                "transport": "sse",
                "url": conn["config"].get("url", ""),
                "headers": conn["config"].get("headers", {}),
                "timeout": conn.get("timeout", 30.0),
                "sse_read_timeout": conn.get("sse_read_timeout", 60.0)
            }
    
    agent_executor = PlatformToolsRunnable.create_agent_executor(
        agent_type="platform-knowledge-mode",
        agents_registry=agents_registry,
        llm=llm,
        tools=tools,
        history=history,
        intermediate_steps=intermediate_steps,
        mcp_connections=mcp_connections if use_mcp else {}
    )

    full_chain = {"chat_input": lambda x: x["input"]} | agent_executor
  
    return full_chain, agent_executor


async def chat(
        query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
        metadata: dict = Body({}, description="附件，可能是图像或者其他功能", examples=[]),
        conversation_id: str = Body("", description="对话框ID"),
        message_id: str = Body(None, description="数据库消息ID"),
        history_len: int = Body(-1, description="从数据库中取历史消息的数量"),
        stream: bool = Body(True, description="流式输出"),
        chat_model_config: dict = Body({}, description="LLM 模型配置", examples=[]),
        tool_config: dict = Body({}, description="工具配置", examples=[]),
        use_mcp: bool = Body(False, description="使用MCP"),
        max_tokens: int = Body(None, description="LLM最大token数配置", example=4096),
):
    """Agent 对话"""

    
    async def chat_iterator_event() -> AsyncIterable[OpenAIChatOutput]:
        try:
            callbacks = []

            # Enable langchain-chatchat to support langfuse
            import os
            # 、、下面这个if是做数据监控用的
            langfuse_secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
            langfuse_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
            langfuse_host = os.environ.get("LANGFUSE_HOST")
            if langfuse_secret_key and langfuse_public_key and langfuse_host:
                from langfuse import Langfuse
                from langfuse.callback import CallbackHandler

                langfuse_handler = CallbackHandler()
                callbacks.append(langfuse_handler)
 
            # 、、获取模型字典和提示词
            models, prompts = create_models_from_config(
                callbacks=callbacks, configs=chat_model_config, stream=stream, max_tokens=max_tokens
            )
            # 、、获取所有工具实例
            all_tools = get_tool().values()
            # 、、工具如果有配置就把这个工具实例捞出来
            tools = [tool for tool in all_tools if tool.name in tool_config]
            # 、、对筛选出来的工具实例进行赋值，并且更新callback属性为传入的callbacks，
            # 、、目的值为了在工具执行的时候使用当前的回调函数，比如记录日志，监控等
            tools = [t.copy(update={"callbacks": callbacks}) for t in tools]
            # 、、
            full_chain, agent_executor = create_models_chains(
                prompts=prompts,
                models=models,
                conversation_id=conversation_id,
                tools=tools,
                callbacks=callbacks,
                history_len=history_len,
                metadata=metadata,
                use_mcp = use_mcp
            )
            message_id = add_message_to_db(
                    chat_type="llm_chat",
                    query=query,
                    conversation_id=conversation_id,
            )
            chat_iterator = full_chain.invoke({
                "input": query
            })
            last_tool = {}
            async for item in chat_iterator:

                data = {}

                data["status"] = item.status
                data["tool_calls"] = []
                data["message_type"] = MsgType.TEXT
                if isinstance(item, PlatformToolsAction):
                    logger.info("PlatformToolsAction:" + str(item.to_json()))
                    data["text"] = item.log
                    tool_call = {
                        "index": 0,
                        "id": item.run_id,
                        "type": "function",
                        "function": {
                            "name": item.tool,
                            "arguments": item.tool_input,
                        },
                        "tool_output": None,
                        "is_error": False,
                    }
                    data["tool_calls"].append(tool_call)

                elif isinstance(item, PlatformToolsFinish):
                    data["text"] = item.log

                    last_tool.update(
                        tool_output=item.return_values["output"],
                    )
                    data["tool_calls"].append(last_tool)

                    try:
                        tool_output = json.loads(item.return_values["output"])
                        if message_type := tool_output.get("message_type"):
                            data["message_type"] = message_type
                    except:
                        ...

                elif isinstance(item, PlatformToolsActionToolStart):
                    logger.info("PlatformToolsActionToolStart:" + str(item.to_json()))

                    last_tool = {
                        "index": 0,
                        "id": item.run_id,
                        "type": "function",
                        "function": {
                            "name": item.tool,
                            "arguments": item.tool_input,
                        },
                        "tool_output": None,
                        "is_error": False,
                    }
                    data["tool_calls"].append(last_tool)

                elif isinstance(item, PlatformToolsActionToolEnd):
                    logger.info("PlatformToolsActionToolEnd:" + str(item.to_json()))
                    last_tool.update(
                        tool_output=item.tool_output,
                        is_error=False,
                    )
                    data["tool_calls"] = [last_tool]

                    last_tool = {}
                    try:
                        tool_output = json.loads(item.tool_output)
                        if message_type := tool_output.get("message_type"):
                            data["message_type"] = message_type
                    except:
                        ...
                elif isinstance(item, PlatformToolsLLMStatus):

                    data["text"] = item.text

                ret = OpenAIChatOutput(
                    id=f"chat{uuid.uuid4()}",
                    object="chat.completion.chunk",
                    content=data.get("text", ""),
                    role="assistant",
                    tool_calls=data["tool_calls"],
                    model=models["llm_model"].model_name,
                    status=data["status"],
                    message_type=data["message_type"],
                    message_id=message_id,
                    class_name=item.class_name()
                )
                yield ret.model_dump_json()

            string_intermediate_steps = dumps(agent_executor.intermediate_steps, pretty=True)

            update_message(
                message_id, 
                agent_executor.history[-1].get("content"),
                metadata = {
                    "intermediate_steps": string_intermediate_steps
                }
            )
             
        except asyncio.exceptions.CancelledError:
            logger.warning("streaming progress has been interrupted by user.")
            return
        except Exception as e:
            logger.error(f"error in chat: {e}")
            yield {"data": json.dumps({"error": str(e)})}
            return

    if stream:
        return EventSourceResponse(chat_iterator_event())
    else:
        ret = OpenAIChatOutput(
            id=f"chat{uuid.uuid4()}",
            object="chat.completion",
            content="",
            role="assistant",
            finish_reason="stop",
            tool_calls=[],
            status=AgentStatus.agent_finish,
            message_type=MsgType.TEXT,
            message_id=message_id,
        )

        async for chunk in chat_iterator_event():
            data = json.loads(chunk)
            if text := data["choices"][0]["delta"]["content"]:
                ret.content += text
            if data["status"] == AgentStatus.tool_end:
                ret.tool_calls += data["choices"][0]["delta"]["tool_calls"]
            ret.model = data["model"]
            ret.created = data["created"]

        return ret.model_dump()