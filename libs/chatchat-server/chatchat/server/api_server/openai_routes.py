from __future__ import annotations

import asyncio
import base64
import os
import shutil
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Dict, Iterable, Tuple

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse
from openai import AsyncClient
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from chatchat.settings import Settings
from chatchat.server.utils import get_config_platforms, get_model_info, get_OpenAIClient
from chatchat.utils import build_logger

from .api_schemas import *

logger = build_logger()


DEFAULT_API_CONCURRENCIES = 5  # 默认单个模型最大并发数
model_semaphores: Dict[
    Tuple[str, str], asyncio.Semaphore
] = {}  # key: (model_name, platform)
openai_router = APIRouter(prefix="/v1", tags=["OpenAI 兼容平台整合接口"])


@asynccontextmanager
async def get_model_client(model_name: str) -> AsyncGenerator[AsyncClient]:
    """
    对重名模型进行调度，依次选择：空闲的模型 -> 当前访问数最少的模型
    """
    max_semaphore = 0
    selected_platform = ""
    model_infos = get_model_info(model_name=model_name, multiple=True)
    assert model_infos, f"specified model '{model_name}' cannot be found in MODEL_PLATFORMS."

    for m, c in model_infos.items():
        key = (m, c["platform_name"])
        api_concurrencies = c.get("api_concurrencies", DEFAULT_API_CONCURRENCIES)
        if key not in model_semaphores:
            model_semaphores[key] = asyncio.Semaphore(api_concurrencies)
        semaphore = model_semaphores[key]
        if semaphore._value >= api_concurrencies:
            selected_platform = c["platform_name"]
            break
        elif semaphore._value > max_semaphore:
            selected_platform = c["platform_name"]

    key = (m, selected_platform)
    semaphore = model_semaphores[key]
    try:
        await semaphore.acquire()
        yield get_OpenAIClient(platform_name=selected_platform, is_async=True)
    except Exception:
        logger.exception(f"failed when request to {key}")
    finally:
        semaphore.release()


async def openai_request(
    method, body, extra_json: Dict = {}, header: Iterable = [], tail: Iterable = []
):
    """
    helper function to make openai request with extra fields
    """

    async def generator():
        try:
            for x in header:
                if isinstance(x, str):
                    x = OpenAIChatOutput(content=x, object="chat.completion.chunk")
                elif isinstance(x, dict):
                    x = OpenAIChatOutput.model_validate(x)
                else:
                    raise RuntimeError(f"unsupported value: {header}")
                for k, v in extra_json.items():
                    setattr(x, k, v)
                yield x.model_dump_json()

            async for chunk in await method(**params):
                for k, v in extra_json.items():
                    setattr(chunk, k, v)
                yield chunk.model_dump_json()

            for x in tail:
                if isinstance(x, str):
                    x = OpenAIChatOutput(content=x, object="chat.completion.chunk")
                elif isinstance(x, dict):
                    x = OpenAIChatOutput.model_validate(x)
                else:
                    raise RuntimeError(f"unsupported value: {tail}")
                for k, v in extra_json.items():
                    setattr(x, k, v)
                yield x.model_dump_json()
        except asyncio.exceptions.CancelledError:
            logger.warning("streaming progress has been interrupted by user.")
            return
        except Exception as e:
            logger.error(f"openai request error: {e}")
            yield {"data": json.dumps({"error": str(e)})}

    params = body.model_dump(exclude_unset=True)
    if params.get("max_tokens") == 0:
        params["max_tokens"] = Settings.model_settings.MAX_TOKENS

    if hasattr(body, "stream") and body.stream:
        return EventSourceResponse(generator())
    else:
        result = await method(**params)
        for k, v in extra_json.items():
            setattr(result, k, v)
        return result.model_dump()


@openai_router.get("/models")
async def list_models() -> Dict:
    """
    整合所有平台的模型列表。
    """

    async def task(name: str, config: Dict):
        try:
            client = get_OpenAIClient(name, is_async=True)
            models = await client.models.list()
            return [{**x.model_dump(), "platform_name": name} for x in models.data]
        except Exception:
            logger.exception(f"failed request to platform: {name}")
            return []

    result = []
    tasks = [
        asyncio.create_task(task(name, config))
        for name, config in get_config_platforms().items()
    ]
    for t in asyncio.as_completed(tasks):
        result += await t

    return {"object": "list", "data": result}


@openai_router.post("/chat/completions")
async def create_chat_completions(
    body: OpenAIChatInput,
):
    async with get_model_client(body.model) as client:
        result = await openai_request(client.chat.completions.create, body)
        return result


@openai_router.post("/completions")
async def create_completions(
    request: Request,
    body: OpenAIChatInput,
):
    async with get_model_client(body.model) as client:
        return await openai_request(client.completions.create, body)


@openai_router.post("/embeddings")
async def create_embeddings(
    request: Request,
    body: OpenAIEmbeddingsInput,
):
    params = body.model_dump(exclude_unset=True)
    client = get_OpenAIClient(model_name=body.model)
    return (await client.embeddings.create(**params)).model_dump()


@openai_router.post("/images/generations")
async def create_image_generations(
    request: Request,
    body: OpenAIImageGenerationsInput,
):
    async with get_model_client(body.model) as client:
        return await openai_request(client.images.generate, body)


@openai_router.post("/images/variations")
async def create_image_variations(
    request: Request,
    body: OpenAIImageVariationsInput,
):
    async with get_model_client(body.model) as client:
        return await openai_request(client.images.create_variation, body)


@openai_router.post("/images/edit")
async def create_image_edit(
    request: Request,
    body: OpenAIImageEditsInput,
):
    async with get_model_client(body.model) as client:
        return await openai_request(client.images.edit, body)


@openai_router.post("/audio/translations", deprecated="暂不支持")
async def create_audio_translations(
    request: Request,
    body: OpenAIAudioTranslationsInput,
):
    async with get_model_client(body.model) as client:
        return await openai_request(client.audio.translations.create, body)


@openai_router.post("/audio/transcriptions", deprecated="暂不支持")
async def create_audio_transcriptions(
    request: Request,
    body: OpenAIAudioTranscriptionsInput,
):
    async with get_model_client(body.model) as client:
        return await openai_request(client.audio.transcriptions.create, body)


@openai_router.post("/audio/speech", deprecated="暂不支持")
async def create_audio_speech(
    request: Request,
    body: OpenAIAudioSpeechInput,
):
    async with get_model_client(body.model) as client:
        return await openai_request(client.audio.speech.create, body)


def _get_file_id(
    purpose: str,
    created_at: int,
    filename: str,
) -> str:
    """、、生成一个文件ID，
    将文件的一些元数据组合成一个字符串然后经过base64编码，以确保不包含特殊字符串，同时可以安全的用作URL中的文件标识符（唯一id）
    1、传入purpose（用途）、create_at(创建时间戳)和filename（文件名），组合成一个字符串，
        格式为 {purpose}/{today}/{filename}, today,使用通过create_at 时间戳转换为%Y-%m-%d格式的日期字符串得到
    2、将这个字符串进行base64 URL安全编码（使用urlsafe_b64encode），然后解码为字符串返回。
        例如，如果purpose是"assistants"，created_at是某个时间戳（对应日期为2023-10-10），filename是"image.png"，那么组合的字符串就是："assistants/2023-10-10/image.png"。
        然后进行base64编码，得到的结果类似于：YXNzaXN0YW50cy8yMDIzLTEwLTEwL2ltYWdlLnBuZw==。
        这样的文件ID可以安全地用在URL中，因为base64 URL安全编码不会包含URL中需要转义的字符。
        同时，这个文件ID包含了文件的元信息，在需要的时候可以通过解码还原出原始路径，以便于文件的管理和查找
    """
    today = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d")
    return base64.urlsafe_b64encode(f"{purpose}/{today}/{filename}".encode()).decode()


def _get_file_info(file_id: str) -> Dict:
    splits = base64.urlsafe_b64decode(file_id).decode().split("/")
    created_at = -1
    size = -1
    file_path = _get_file_path(file_id)
    if os.path.isfile(file_path):
        created_at = int(os.path.getmtime(file_path))
        size = os.path.getsize(file_path)

    return {
        "purpose": splits[0],
        "created_at": created_at,
        "filename": splits[2],
        "bytes": size,
    }


def _get_file_path(file_id: str) -> str:
    file_id = base64.urlsafe_b64decode(file_id).decode()
    return os.path.join(Settings.basic_settings.BASE_TEMP_DIR, "openai_files", file_id)


@openai_router.post("/files")
async def files(
    request: Request,
    file: UploadFile,
    purpose: str = "assistants",
) -> Dict:
    # 、、计算一个时间戳
    created_at = int(datetime.now().timestamp())
    # 、、根据传进去的元数据，生成一个base64的唯一file_id
    file_id = _get_file_id(
        purpose=purpose, created_at=created_at, filename=file.filename
    )
    # 、、file_id中带有path的信息，事实上就是通过上面的元数据生成的一个path的base64编码的
    file_path = _get_file_path(file_id)
    # 、、创建存储目录
    file_dir = os.path.dirname(file_path) # 提取文件路径的目录部分
    os.makedirs(file_dir, exist_ok=True) # 递归创建目录，

    with open(file_path, "wb") as fp: # 以二进制写入模式打开文件
        #、、高效将上传的内容赋值到目标文件，使用分块复制，避免内存溢出
        # 、、file.file是上传的文件对象，
        # 、、fp是目标文件的文件对象
        # 、、重点： shutil.copyfileobj这个函数会从源文件中读取数据
        # 并写入到目标文件fp中。它默认使用分块复制， 即一次读取一定大小的数据，然后写入，然后在读取一块直到其全部复制完成
        shutil.copyfileobj(file.file, fp) 
    # 、、释放上传文件的资源
    file.file.close()

    # 、、将信息返回
    return dict(
        id=file_id,
        filename=file.filename,
        bytes=file.size, 
        created_at=created_at,
        object="file",
        purpose=purpose,
    )


@openai_router.get("/files")
def list_files(purpose: str) -> Dict[str, List[Dict]]:
    file_ids = []
    root_path = Path(Settings.basic_settings.BASE_TEMP_DIR) / "openai_files" / purpose
    for dir, sub_dirs, files in os.walk(root_path):
        dir = Path(dir).relative_to(root_path).as_posix()
        for file in files:
            file_id = base64.urlsafe_b64encode(
                f"{purpose}/{dir}/{file}".encode()
            ).decode()
            file_ids.append(file_id)
    return {
        "data": [{**_get_file_info(x), "id": x, "object": "file"} for x in file_ids]
    }


@openai_router.get("/files/{file_id}")
def retrieve_file(file_id: str) -> Dict:
    file_info = _get_file_info(file_id)
    return {**file_info, "id": file_id, "object": "file"}


@openai_router.get("/files/{file_id}/content")
def retrieve_file_content(file_id: str) -> Dict:
    file_path = _get_file_path(file_id)
    return FileResponse(file_path)


@openai_router.delete("/files/{file_id}")
def delete_file(file_id: str) -> Dict:
    file_path = _get_file_path(file_id)
    deleted = False

    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
            deleted = True
    except:
        ...

    return {"id": file_id, "deleted": deleted, "object": "file"}
