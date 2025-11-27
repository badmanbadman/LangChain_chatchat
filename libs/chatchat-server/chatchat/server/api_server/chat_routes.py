from __future__ import annotations

from typing import Dict, List

from fastapi import APIRouter, Request
from langchain.prompts.prompt import PromptTemplate
from sse_starlette import EventSourceResponse

from chatchat.server.api_server.api_schemas import OpenAIChatInput
from chatchat.server.chat.chat import chat
from chatchat.server.chat.kb_chat import kb_chat
from chatchat.server.chat.feedback import chat_feedback
from chatchat.server.chat.file_chat import file_chat
from chatchat.server.db.repository import add_message_to_db
from chatchat.server.utils import (
    get_OpenAIClient,
    get_prompt_template,
    get_tool,
    get_tool_config,
)
from chatchat.settings import Settings
from chatchat.utils import build_logger
from .openai_routes import openai_request, OpenAIChatOutput


logger = build_logger()

chat_router = APIRouter(prefix="/chat", tags=["ChatChat 对话"])

# chat_router.post(
#     "/chat",
#     summary="与llm模型对话(通过LLMChain)",
# )(chat)

chat_router.post(
    "/feedback",
    summary="返回llm模型对话评分",
)(chat_feedback)


chat_router.post("/kb_chat", summary="知识库对话")(kb_chat)
chat_router.post("/file_chat", summary="文件对话")(file_chat)


@chat_router.post("/chat/completions", summary="兼容 openai 的统一 chat 接口")
async def chat_completions(
    request: Request,
    body: OpenAIChatInput,
) -> Dict:
    """
    请求参数与 openai.chat.completions.create 一致，可以通过 extra_body 传入额外参数
    tools 和 tool_choice 可以直接传工具名称，会根据项目里包含的 tools 进行转换
    通过不同的参数组合调用不同的 chat 功能：
    - tool_choice
        - extra_body 中包含 tool_input: 直接调用 tool_choice(tool_input)
        - extra_body 中不包含 tool_input: 通过 agent 调用 tool_choice
    - tools: agent 对话
    - 其它：LLM 对话
    以后还要考虑其它的组合（如文件对话）
    返回与 openai 兼容的 Dict
    """
    # import rich
    # rich.print(body)

    # 当调用本接口且 body 中没有传入 "max_tokens" 参数时, 默认使用配置中定义的值
    if body.max_tokens in [None, 0]:
        body.max_tokens = Settings.model_settings.MAX_TOKENS

    client = get_OpenAIClient(model_name=body.model, is_async=True)

    """、、model_extra:
    model_extra 的工作原理:
        当 Pydantic 模型配置了 extra = "allow" 时：
        标准字段：会被正常验证和解析
        额外字段：会被收集到 model_extra 属性中
    客户端请求示例：
        {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true,
            "max_tokens": 1000,
            "conversation_id": "conv_123",        // 额外字段
            "tool_input": {"param": "value"},     // 额外字段
            "use_mcp": false,                     // 额外字段
            "chat_model_config": {"temperature": 0.7}  // 额外字段
        }
    后端解析结果：
        # 标准字段被正常解析
        body.model = "gpt-4"
        body.messages = [{"role": "user", "content": "Hello"}]
        body.stream = True
        body.max_tokens = 1000

        # 额外字段进入 model_extra
        body.model_extra = {
            "conversation_id": "conv_123",
            "tool_input": {"param": "value"},
            "use_mcp": false,
            "chat_model_config": {"temperature": 0.7}
        }
    """
    # 提取所有额外参数
    extra = {**body.model_extra} or {}
    for key in list(extra): # 从 body 中删除这些额外属性，避免干扰后续处理
        delattr(body, key) # 删除 body.conversation_id, body.tool_input 等

    # 现在 body 只包含标准的 OpenAI 参数
    # extra 包含所有自定义参数

    # check tools & tool_choice in request body
    if isinstance(body.tool_choice, str):
        # 、、如果传入的tool_choice是工具名称（字符串），则通过工具名称获取到工具实例
        if t := get_tool(body.tool_choice):
            # 、、根据工具实例再次组装数据，转换为标准格式
            body.tool_choice = {"function": {"name": t.name}, "type": "function"}
    if isinstance(body.tools, list):
        # 、、如果传入的是tools的List（是个名称组成的List）
        for i in range(len(body.tools)):
            if isinstance(body.tools[i], str):
                # 、、根据名称获取工具实例
                if t := get_tool(body.tools[i]):
                    # 、、组装数据，组装的数据格式和上面组装tool_choice一样
                    body.tools[i] = {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.args,
                        },
                    }

    # 、、会话ID，前端传来的
    conversation_id = extra.get("conversation_id")
  
    try:
        # 、、聊天消息储存到数据库
        message_id = (
            add_message_to_db(
                chat_type="agent_chat",
                query=body.messages[-1]["content"], # 最新的用户发来的消息
                conversation_id=conversation_id,
            )
            if conversation_id
            else None
        )
    except Exception as e:
        logger.warning(f"failed to add message to db: {e}")
        message_id = None

    chat_model_config = {}  # TODO: 前端支持配置模型
    tool_config = {}
    if body.tools:
        # 、、将tool的name字段捞出来
        tool_names = [x["function"]["name"] for x in body.tools]
        # 、、根据工具名称，获取配置工具信息
        tool_config = {name: get_tool_config(name) for name in tool_names}

    result = await chat(
        query=body.messages[-1]["content"], # 用户查询
        metadata=extra.get("metadata", {}), #元数据
        conversation_id=extra.get("conversation_id", ""), #会话ID
        message_id=message_id,#消息Id
        history_len=-1, #使用所有历史
        stream=body.stream, #流式输出
        chat_model_config=extra.get("chat_model_config", chat_model_config), #聊天模型配置
        tool_config=tool_config, #工具配置
        use_mcp=extra.get("use_mcp", False), #mac开关
        max_tokens=body.max_tokens, #token限制
    )
    return result
