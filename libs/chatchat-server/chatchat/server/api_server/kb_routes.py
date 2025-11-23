from __future__ import annotations

from typing import List, Literal

from fastapi import APIRouter, Request

from chatchat.settings import Settings
from chatchat.server.api_server.api_schemas import OpenAIChatInput, OpenAIChatOutput
from chatchat.server.chat.file_chat import upload_temp_docs
from chatchat.server.chat.kb_chat import kb_chat
from chatchat.server.knowledge_base.kb_api import create_kb, delete_kb, list_kbs
from chatchat.server.knowledge_base.kb_doc_api import (
    delete_docs,
    download_doc,
    list_files,
    recreate_vector_store,
    search_docs,
    update_docs,
    update_info,
    upload_docs,
    search_temp_docs,
)
from chatchat.server.knowledge_base.kb_summary_api import (
    recreate_summary_vector_store,
    summary_doc_ids_to_vector_store,
    summary_file_to_vector_store,
)
from chatchat.server.utils import BaseResponse, ListResponse
from chatchat.server.knowledge_base.kb_cache.faiss_cache import memo_faiss_pool

# 、、API路由器，用于管理知识库相关的API路由
kb_router = APIRouter(prefix="/knowledge_base", tags=["Knowledge Base Management"])


# 、、在这个端点会将前端使用的 非标准openai.Client (自定义了base_url)接口调用,从中接收到的mode和param参数解析出来，
@kb_router.post(
    "/{mode}/{param}/chat/completions", summary="知识库对话，openai 兼容，参数与 /chat/kb_chat 一致"
)
async def kb_chat_endpoint(
    mode: Literal["local_kb", "temp_kb", "search_engine"],
    param: str,
    body: OpenAIChatInput,
    request: Request,
):
    # import rich
    # rich.print(body)

    if body.max_tokens in [None, 0]:
        # 填充默认最大token数
        body.max_tokens = Settings.model_settings.MAX_TOKENS

    # 、、获取前端传过来的额外参数
    extra = body.model_extra
    # 、、组装参数，调用kb_chat，kb_chat中根据不同的mode进行不同类型的处理，（本地知识库，临时知识库，搜索引擎）
    ret = await kb_chat(
        query=body.messages[-1]["content"], # 、、用户的最新输入，（最后一条数据）
        mode=mode, #、、知识来源
        kb_name=param,
        top_k=extra.get("top_k", Settings.kb_settings.VECTOR_SEARCH_TOP_K),
        score_threshold=extra.get("score_threshold", Settings.kb_settings.SCORE_THRESHOLD),# 、、知识分数阈值
        history=body.messages[:-1], # 、、历史对话记录（默认第一条到倒数第二条）
        stream=body.stream, # 、、是否流式输出，默认为False，需要前端调用openaiClient的时候显示的指的为True才会流式输出
        model=body.model, #、、使用的模型名称
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        prompt_name=extra.get("prompt_name", "default"),
        return_direct=extra.get("return_direct", False),
        request=request, # 对象包含了完整的 HTTP 请求信息，可以用于进行身份授权和限流，日志等，虽然kb_chat中暂时没用到
    )
    return ret


kb_router.get(
    "/list_knowledge_bases", response_model=ListResponse, summary="获取知识库列表"
)(list_kbs)

kb_router.post(
    "/create_knowledge_base", response_model=BaseResponse, summary="创建知识库"
)(create_kb)

kb_router.post(
    "/delete_knowledge_base", response_model=BaseResponse, summary="删除知识库"
)(delete_kb)

kb_router.get(
    "/list_files", response_model=ListResponse, summary="获取知识库内的文件列表"
)(list_files)

kb_router.post("/search_docs", response_model=List[dict], summary="搜索知识库")(
    search_docs
)

kb_router.post(
    "/upload_docs",
    response_model=BaseResponse,
    summary="上传文件到知识库，并/或进行向量化",
)(upload_docs)

kb_router.post(
    "/delete_docs", response_model=BaseResponse, summary="删除知识库内指定文件"
)(delete_docs)

kb_router.post("/update_info", response_model=BaseResponse, summary="更新知识库介绍")(
    update_info
)

kb_router.post(
    "/update_docs", response_model=BaseResponse, summary="更新现有文件到知识库"
)(update_docs)

kb_router.get("/download_doc", summary="下载对应的知识文件")(download_doc)

kb_router.post(
    "/recreate_vector_store", summary="根据content中文档重建向量库，流式输出处理进度。"
)(recreate_vector_store)

kb_router.post("/upload_temp_docs", summary="上传文件到临时目录，用于文件对话。")(
    upload_temp_docs
)

kb_router.post("/search_temp_docs", summary="检索临时知识库")(
    search_temp_docs
)

# @kb_router.post("/list_temp_kbs", summary="列出所有临时知识库")
# def list_temp_kbs():
#     return list(memo_faiss_pool.keys())


summary_router = APIRouter(prefix="/kb_summary_api")
summary_router.post(
    "/summary_file_to_vector_store", summary="单个知识库根据文件名称摘要"
)(summary_file_to_vector_store)
summary_router.post(
    "/summary_doc_ids_to_vector_store",
    summary="单个知识库根据doc_ids摘要",
    response_model=BaseResponse,
)(summary_doc_ids_to_vector_store)
summary_router.post("/recreate_summary_vector_store", summary="重建单个知识库文件摘要")(
    recreate_summary_vector_store
)

kb_router.include_router(summary_router)
