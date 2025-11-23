from __future__ import annotations

import asyncio, json
import uuid
from typing import AsyncIterable, List, Optional, Literal

from fastapi import Body, Request
from fastapi.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.prompts.chat import ChatPromptTemplate


from chatchat.settings import Settings
from chatchat.server.agent.tools_factory.search_internet import search_engine
from chatchat.server.api_server.api_schemas import OpenAIChatOutput
from chatchat.server.chat.utils import History
from chatchat.server.knowledge_base.kb_service.base import KBServiceFactory
from chatchat.server.knowledge_base.kb_doc_api import search_docs, search_temp_docs
from chatchat.server.knowledge_base.utils import format_reference
from chatchat.server.utils import (wrap_done, get_ChatOpenAI, get_default_llm,
                                   BaseResponse, get_prompt_template, build_logger,
                                   check_embed_model, api_address
                                )


logger = build_logger()


async def kb_chat(
                query: str = Body(..., description="用户输入", examples=["你好"]),
                mode: Literal["local_kb", "temp_kb", "search_engine"] = Body("local_kb", description="知识来源"),
                kb_name: str = Body("", description="mode=local_kb时为知识库名称；temp_kb时为临时知识库ID，search_engine时为搜索引擎名称", examples=["samples"]),
                top_k: int = Body(Settings.kb_settings.VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                score_threshold: float = Body(
                    Settings.kb_settings.SCORE_THRESHOLD,
                    description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                    ge=0,
                    le=2,
                ),
                history: List[History] = Body(
                    [],
                    description="历史对话",
                    examples=[[
                        {"role": "user",
                        "content": "我们来玩成语接龙，我先来，生龙活虎"},
                        {"role": "assistant",
                        "content": "虎头虎脑"}]]
                ),
                stream: bool = Body(True, description="流式输出"),
                model: str = Body(get_default_llm(), description="LLM 模型名称。"),
                temperature: float = Body(Settings.model_settings.TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
                max_tokens: Optional[int] = Body(
                    Settings.model_settings.MAX_TOKENS,
                    description="限制LLM生成Token数量，默认None代表模型最大值"
                ),
                prompt_name: str = Body(
                    "default",
                    description="使用的prompt模板名称(在prompt_settings.yaml中配置)"
                ),
                return_direct: bool = Body(False, description="直接返回检索结果，不送入 LLM"),
                request: Request = None,
                ):
    if mode == "local_kb":
        kb = KBServiceFactory.get_service_by_name(kb_name)
        # 、、没找到知识库直接报404
        if kb is None:
            return BaseResponse(code=404, msg=f"未找到知识库 {kb_name}")
    
    async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
        try:
            nonlocal history, prompt_name, max_tokens

            # 、、将历史对话转换为 History 对象列表，前端传来的是List[{"role": "user", "content": prompt}]这种格式（字典列表）
            # 、、将字典传入转化函数，返回一个History对象
            history = [History.from_data(h) for h in history]

            if mode == "local_kb":
                kb = KBServiceFactory.get_service_by_name(kb_name)
                ok, msg = kb.check_embed_model()
                if not ok:
                    # 嵌入模型不可用，手动抛错
                    raise ValueError(msg)
                # 在线程池批量 进行 search_docs 处理，入参是后面 后面的这一堆堆
                # search_docs进行了混合检索（关键词和向量检索权重各占50%，进行融合后的结果）
                docs = await run_in_threadpool(search_docs,
                                                query=query,
                                                knowledge_base_name=kb_name,
                                                top_k=top_k,
                                                score_threshold=score_threshold,
                                                file_name="",
                                                metadata={})
                # 对检索的dos进行格式化
                source_documents = format_reference(kb_name, docs, api_address(is_public=True))
            elif mode == "temp_kb":
                ok, msg = check_embed_model()
                if not ok:
                    raise ValueError(msg)
                # search_temp_docs 直接进行向量检索
                docs = await run_in_threadpool(search_temp_docs,
                                                kb_name,
                                                query=query,
                                                top_k=top_k,
                                                score_threshold=score_threshold)
                # 格式化检索结果
                source_documents = format_reference(kb_name, docs, api_address(is_public=True))
            elif mode == "search_engine":
                result = await run_in_threadpool(search_engine, query, top_k, kb_name)
                # 、、将返回结果的docs取出并重新组装为字典数组
                docs = [x.dict() for x in result.get("docs", [])]
                # 格式化
                source_documents = [f"""出处 [{i + 1}] [{d['metadata']['filename']}]({d['metadata']['source']}) \n\n{d['page_content']}\n\n""" for i,d in enumerate(docs)]
            else:
                docs = []
                source_documents = []
            # import rich
            # rich.print(dict(
            #     mode=mode,
            #     query=query,
            #     knowledge_base_name=kb_name,
            #     top_k=top_k,
            #     score_threshold=score_threshold,
            # ))
            # rich.print(docs)

            # 、、如果，是仅返回检索结果，直接进行数据组装
            if return_direct:
                yield OpenAIChatOutput(
                    id=f"chat{uuid.uuid4()}", # 生成唯一对话id
                    model=None,# 没有使用LLM模型
                    object="chat.completion", #OpenAPO兼容格式
                    content="", # 内容为空（没有LLM生成）
                    role="assistant", # 角色为助手
                    finish_reason="stop", #完成原因
                    docs=source_documents, #关键：：直接返回检索到的文档
                ) .model_dump_json() # 将OpenAIChatOutput对象转为JSON字符串
                return #直接返回，不执行后续LLM生成

            # 、、LangChain中的回调处理器，
            # 主要是为了一些用户问题，性能，日志等等的监听做数据准备
            callback = AsyncIteratorCallbackHandler()
            """
            callback这个实例值得详细说下：
                class AsyncIteratorCallbackHandler(BaseCallbackHandler):
                    # 1、类初始化方法
                    def __init__(self):
                        self.queue = asyncio.Queue()  # 异步队列，用于存储生成的令牌
                        self.done = asyncio.Event() #完成事件，用于通知生成结束
                        self._lock = asyncio.Lock() #锁，确保线程安全
                        self._tokens = [] # 可选： 存储所有令牌（大致看来下源码，这个属性没有）
                    
                    # 2 关键的异步迭代器方法,（允许使用asyn for 循环消费令牌） 
                    async def aiter(self):
                        while not self.done.is_set() or not self.queue.empty():
                            # 等待队列中有新令牌或生成完成
                            try：
                                # 从队列中获取令牌，最多等待1秒
                                token = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                                yield token
                                self.queue.task_done() # 标记任务完成
                            except asyncio.TimeoutError:
                                # 超时但生成还未完成，继续等待
                                if self.done.is_set():
                                    break
                                continue
                            except Exception as e:
                                logger.error(e)
            执行时序图
            LLM 生成过程                AsyncIteratorCallbackHandler             消费代码 (async for)
                |                               |                                      |
                |--- on_llm_start() ----------->|                                      |
                |                               |                                      |
                |--- on_llm_new_token("Hello") -> queue.put("Hello")                   |
                |                               |--- queue.get() --------------------->| yield "Hello"
                |                               |                                      |
                |--- on_llm_new_token(" world") -> queue.put(" world")                |
                |                               |--- queue.get() --------------------->| yield " world"
                |                               |                                      |
                |--- on_llm_new_token("!") -----> queue.put("!")                       |
                |                               |--- queue.get() --------------------->| yield "!"
                |                               |                                      |
                |--- on_llm_end() ------------->| set done event                       |
                |                               |--- detect done event --------------->| break loop
            """
            callbacks = [callback]

            # Enable langchain-chatchat to support langfuse
            # 、、 集成 Langfuse 监控和追踪系统 到 Milvus 向量库的应用中，这个langfuse主要就是为监控和追踪的
            import os
            langfuse_secret_key = os.environ.get('LANGFUSE_SECRET_KEY')
            langfuse_public_key = os.environ.get('LANGFUSE_PUBLIC_KEY')
            langfuse_host = os.environ.get('LANGFUSE_HOST')
            if langfuse_secret_key and langfuse_public_key and langfuse_host :
                from langfuse import Langfuse  
                from langfuse.callback import CallbackHandler
                langfuse_handler = CallbackHandler()
                callbacks.append(langfuse_handler)
            # 、、获取最大tokens长度，优先用传过来的值，
            if max_tokens in [None, 0]:
                max_tokens = Settings.model_settings.MAX_TOKENS

            # 获取聊天模型实例
            llm = get_ChatOpenAI(
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,
                callbacks=callbacks,
            )
            # TODO： 视情况使用 API
            # # 加入reranker
            # if Settings.kb_settings.USE_RERANKER:
            #     reranker_model_path = get_model_path(Settings.kb_settings.RERANKER_MODEL)
            #     reranker_model = LangchainReranker(top_n=top_k,
            #                                     device=embedding_device(),
            #                                     max_length=Settings.kb_settings.RERANKER_MAX_LENGTH,
            #                                     model_name_or_path=reranker_model_path
            #                                     )
            #     print("-------------before rerank-----------------")
            #     print(docs)
            #     docs = reranker_model.compress_documents(documents=docs,
            #                                              query=query)
            #     print("------------after rerank------------------")
            #     print(docs)

            # 格式化从知识库中获取的内容，拼接为字符串，并且用 换行 隔开
            context = "\n\n".join([doc["page_content"] for doc in docs])

            if len(docs) == 0: 
                # 如果没有找到相关文档，使用empty模板（如果docs有，就用default模板）
                prompt_name = "empty"
            # 、、rag 提示词模板 
            prompt_template = get_prompt_template("rag", prompt_name)
            # 1. 将当前用户输入转换为消息模板
            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            # 2. 构建完整的聊天提示模板
            chat_prompt = ChatPromptTemplate.from_messages(
                [i.to_msg_template() for i in history] + [input_msg]) # 历史消息 + 当前输入

            chain = chat_prompt | llm
            """、、| 管道操作符
                这里使用LangChain的表达式语言LCEL，创建了一个链，该链首先传递给chat_prompt（一个提升模板），
                然后传递给llm（语言模型），这相当于一个简单链，先格式化提示，再调用模型。
            """

            # Begin a task that runs in the background.
            task = asyncio.create_task(
                wrap_done(
                    chain.ainvoke({"context": context, "question": query}),
                    callback.done
                ),
            )
            """、、task 创建任务
            asyncio.create_task 创建了一个异步任务，这个任务会调用链(chain)的ainvoke方法（异步调用），
            并且传递以一个字典作为输入。这个字典包括两个键，context和question，分别对应提示模板中的变量。

            另外这里使用了以恶搞wrap_done函数，它通常用于在异步任务完成时候设置一个事件（callback.done）,
            wrap_done 函数会等待异步调用完成，然后设置事件，这样外部就可以知道任务已经完成（callback.done 一般是一个 asyncio.Event）

            通常wrap_done函数定义如下
                async def wrap_done(fn: Awaitable, event: asyncio.Event):
                    try: 
                        await fn
                    finally:
                        event.set()
            当链的异步调用完成（无论成功还是失败）时，callback.done事件都会被设置。

            在这个场景中我们有个异步迭代回调（AsyncIteratorCallbackHandler）正在使用，这个异步迭代回调，有个done事件来通知流式传输的结束
            这段代码的整体作用是，在后台异步调用链，并且在链调用完成时候设置done事件，以便通知其他部分。

            这个task是在后台运行的，主程序不会被阻断的。通常我们会一边生成令牌，一边等待任务完成
            
            数据流转： {'context':context, 'question': query}是如何流转的呢？
                当调用chain.ainvoke时，实际上会依次调用chat_prompt和llm
                具体步骤如下：
                    1、chain.ainvoke(input) 首先将input传递给chat_prompt.ainvoke(input) 
                    2、chat_prompt.ainvoke(input)会使用input字典来格式化提示模板，生成一个PromptValue对象（通常是ChatPromptValue，包含了一系列Message对象）
                    3、然后，这个PromptValue对象会传递给llm.ainvoke(PromptValue)
                    4、llm处理这个PromptValue，生成一个LLMResult（或者对于聊天模型可能是ChatMessage）并返回
                注意：这个字典中的属性其实就是我们配置的模板中需要被替换的部分，
            
            
            整体调用链路如下：
                主协程
                    ↓ 调用 asyncio.create_task()
                    ↓ 创建后台任务
                    ↓ 任务开始执行 wrap_done()
                    ↓ 在 wrap_done() 中调用 chain.ainvoke(inputs)  ← 数据在这里传递
                        ↓ 在 chain.ainvoke() 内部：
                            ↓ chat_prompt.ainvoke(inputs)  ← 模板格式化在这里执行
                            ↓ llm.ainvoke(formatted_prompt) ← LLM生成在这里执行
                    ↓ 设置 callback.done 事件
            """
            
            # 、、当通过 知识库 （或者临时文件，或者搜索引擎）没有找到相关的文档时，
            # 将查找结果赋 一段‘未找到文档的’ 文档片段 
            if len(source_documents) == 0:  # 没有找到相关文档
                source_documents.append(f"<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")

            # 、、如果是流式输出 True
            if stream:
                # 、、先返回检索到的文档（一把全返回），这样客户端就能立即收到文档信息了
                ret = OpenAIChatOutput(
                    id=f"chat{uuid.uuid4()}",
                    object="chat.completion.chunk",
                    content="",
                    role="assistant",
                    model=model,
                    docs=source_documents, #、、这个主要返回 检索到的，组装后的文档信息
                )
                yield ret.model_dump_json()
                """
                使用callback.aiter() 逐令牌获取模型生成的内容，没获取一个令牌，就创建一个新的
                OpenAIChatOutput对象（不包含文档，只包含当前令牌），并且yield（返回）其JSON字符串
                callback 这个实例， 在llm中callback，任务链中callback，这里callback都是同一个实例
                因此在llm中进行生成任务的时候调用on_llm_start,on_llm_new_token等方法会这里获取到队列数据
                """
                async for token in callback.aiter():
                    ret = OpenAIChatOutput(
                        id=f"chat{uuid.uuid4()}",
                        object="chat.completion.chunk",
                        content=token,
                        role="assistant",
                        model=model,
                    )
                    yield ret.model_dump_json()
            else:
                # 非流式的直接等callback.aiter执行完毕,全部添加到answer上,
                # 然后转化为json格式一把返回
                answer = ""
                async for token in callback.aiter():
                    answer += token
                ret = OpenAIChatOutput(
                    id=f"chat{uuid.uuid4()}",
                    object="chat.completion",
                    content=answer,
                    role="assistant",
                    model=model,
                )
                yield ret.model_dump_json()
            
            # 、、等待后台任务完成再真正将knowledge_base_chat_iterator这个函数结束
            # 、、资源清理：确保所有资源正确是否
            # 、、错误处理：如果生成过程中出现错误，可以在这里捕获
            await task
        except asyncio.exceptions.CancelledError:
            logger.warning("streaming progress has been interrupted by user.")
            return
        except Exception as e:
            logger.error(f"error in knowledge chat: {e}")
            yield {"data": json.dumps({"error": str(e)})}
            return

    if stream:
        """使用流式返回
        # 、、使用流式返回 用于在FastAPI中实现服务器发送事件（SSE）
        # 、、SSE是一种允许服务器想客户端推送数据的web技术，特点是:
            1、单向通信： 服务器 -> 客户端
            2、基于HTTP协议： 使用标准的HTTP协议
            3、长连接：保持连接打开，持续发送数据
            4、自动重连：客户端断开后会自动尝试重连
            5、文本数据： 主要用于发送文本数据，如JSON、XML等
        # 、、适用场景： 实时通知、聊天应用、动态内容更新
        、、在前端可以使用EventSource API来接收和处理SSE消息
        、、在后端使用SSE可以实现实时数据更新，提升用户体验
        """
        return EventSourceResponse(knowledge_base_chat_iterator())
    else:
        return await knowledge_base_chat_iterator().__anext__()
