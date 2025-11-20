import importlib
import json
import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlencode
from typing import Dict, Generator, List, Tuple, Union

import chardet
import langchain_community.document_loaders
from langchain.docstore.document import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter, TextSplitter
from langchain_community.document_loaders import JSONLoader, TextLoader

from chatchat.settings import Settings
from chatchat.server.file_rag.text_splitter import (
    zh_title_enhance as func_zh_title_enhance,
)
from chatchat.server.utils import run_in_process_pool, run_in_thread_pool
from chatchat.utils import build_logger


logger = build_logger()


def validate_kb_name(knowledge_base_id: str) -> bool:
    # 检查是否包含预期外的字符或路径攻击关键字
    if "../" in knowledge_base_id:
        return False
    return True


def get_kb_path(knowledge_base_name: str):
    return os.path.join(Settings.basic_settings.KB_ROOT_PATH, knowledge_base_name)


def get_doc_path(knowledge_base_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "content")


def get_vs_path(knowledge_base_name: str, vector_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "vector_store", vector_name)


def get_file_path(knowledge_base_name: str, doc_name: str):
    """
    计算并返回知识库中文档的绝对路径（字符串），但只在该路径确实位于知识库的 content 目录下时返回；否则返回 None。
    目的主要是防止路径遍历（../）或传入的 doc_name 指向仓库外的文件。
    """
    doc_path = Path(get_doc_path(knowledge_base_name)).resolve()
    file_path = (doc_path / doc_name).resolve()
    if str(file_path).startswith(str(doc_path)):
        return str(file_path)


def   list_kbs_from_folder():
    """
    、、获取知识库文件夹（knowledge_base）下的所有文件夹，返回一个 List[文件夹名字]
    """
    return [
        f
        for f in os.listdir(Settings.basic_settings.KB_ROOT_PATH)
        if os.path.isdir(os.path.join(Settings.basic_settings.KB_ROOT_PATH, f))
    ]


def list_files_from_folder(kb_name: str):
    """返回文件名称List"""
    doc_path = get_doc_path(kb_name)
    # 所有知识库（文件夹）下的文件存储变量
    result = []

    def is_skiped_path(path: str):
        tail = os.path.basename(path).lower()
        for x in ["temp", "tmp", ".", "~$"]:
            if tail.startswith(x):
                return True
        return False

    def process_entry(entry):
        if is_skiped_path(entry.path):
            return

        if entry.is_symlink():
            target_path = os.path.realpath(entry.path)
            with os.scandir(target_path) as target_it:
                for target_entry in target_it:
                    process_entry(target_entry)
        elif entry.is_file():
            file_path = Path(
                os.path.relpath(entry.path, doc_path)
            ).as_posix()  # 路径统一为 posix 格式
            result.append(file_path)
        elif entry.is_dir():
            with os.scandir(entry.path) as it:
                for sub_entry in it:
                    process_entry(sub_entry)

    with os.scandir(doc_path) as it:
        for entry in it:
            process_entry(entry)

    return result


LOADER_DICT = {
    # 挪个位置，这几个都是使用自定义的加载器
    # "FilteredCSVLoader": [".csv"], 如果使用自定义分割csv  #、、自定义csv加载器，暂时没有用
    "RapidOCRPDFLoader": [".pdf"],  # 、、自定义pdf加载器
    "RapidOCRDocLoader": [".docx"], # 、、自定义docx加载器，
    "RapidOCRPPTLoader": [
        ".ppt",
        ".pptx",
    ],  # 、、自定义ppt,pptx加载器，牛逼啊，这也能自定义的
    "RapidOCRLoader": [".png", ".jpg", ".jpeg", ".bmp"], #、、 自定义图片加载器



    "UnstructuredHTMLLoader": [".html", ".htm"],  
    "MHTMLLoader": [".mhtml"],
    "TextLoader": [".md"],
    "UnstructuredMarkdownLoader": [".md"],
    "JSONLoader": [".json"],
    "JSONLinesLoader": [".jsonl"],
    "CSVLoader": [".csv"],

    "UnstructuredFileLoader": [
        ".eml",
        ".msg",
        ".rst",
        ".rtf",
        ".txt",
        ".xml",
        ".epub",
        ".odt",
        ".tsv",
    ],
    "UnstructuredEmailLoader": [".eml", ".msg"],
    "UnstructuredEPubLoader": [".epub"],
    "UnstructuredExcelLoader": [".xlsx", ".xls", ".xlsd"],
    "NotebookLoader": [".ipynb"],
    "UnstructuredODTLoader": [".odt"],
    "PythonLoader": [".py"],
    "UnstructuredRSTLoader": [".rst"],
    "UnstructuredRTFLoader": [".rtf"],
    "SRTLoader": [".srt"],
    "TomlLoader": [".toml"],
    "UnstructuredTSVLoader": [".tsv"],
    "UnstructuredWordDocumentLoader": [".docx"],
    "UnstructuredXMLLoader": [".xml"],
    "UnstructuredPowerPointLoader": [".ppt", ".pptx"],
    "EverNoteLoader": [".enex"],
}
SUPPORTED_EXTS = [ext for sublist in LOADER_DICT.values() for ext in sublist]


# patch json.dumps to disable ensure_ascii
def _new_json_dumps(obj, **kwargs):
    kwargs["ensure_ascii"] = False
    return _origin_json_dumps(obj, **kwargs)


if json.dumps is not _new_json_dumps:
    _origin_json_dumps = json.dumps
    json.dumps = _new_json_dumps


class JSONLinesLoader(JSONLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._json_lines = True


langchain_community.document_loaders.JSONLinesLoader = JSONLinesLoader


def get_LoaderClass(file_extension):
    for LoaderClass, extensions in LOADER_DICT.items():
        if file_extension in extensions:
            return LoaderClass


def get_loader(loader_name: str, file_path: str, loader_kwargs: Dict = None):
    """
    根据loader_name和文件路径或内容返回文档加载器。
    """
    loader_kwargs = loader_kwargs or {}
    try:
        if loader_name in [
            "RapidOCRPDFLoader",
            "RapidOCRLoader",
            "FilteredCSVLoader",
            "RapidOCRDocLoader",
            "RapidOCRPPTLoader",
        ]:
            #、、 如果是这几种加载器就返回自定义的在file_rag下面的document_loader
            document_loaders_module = importlib.import_module(
                "chatchat.server.file_rag.document_loaders"
            )
        else:
            #、、 如果是其他类型的加载器，就使用langchain_community提供的document_loader加载器
            document_loaders_module = importlib.import_module(
                "langchain_community.document_loaders"
            )
        # 根据加载器的名称，获取加载器
        DocumentLoader = getattr(document_loaders_module, loader_name)
    except Exception as e:
        # 、、按理说到这步来的时候已经在前面校验过文档类型了，不应该报这个错来，后续审查下前端逻辑
        msg = f"为文件{file_path}查找加载器{loader_name}时出错：{e}"
        logger.error(f"{e.__class__.__name__}: {msg}")
        # 、、报错就默认使用langchain_community.document_loader中的加载器
        document_loaders_module = importlib.import_module(
            "langchain_community.document_loaders"
        )
        # 、、报错就默认就给他加载个非结构化的加载器来解析
        DocumentLoader = getattr(document_loaders_module, "UnstructuredFileLoader")

    if loader_name == "UnstructuredFileLoader":
        loader_kwargs.setdefault("autodetect_encoding", True)
    elif loader_name == "CSVLoader":
        if not loader_kwargs.get("encoding"):
            # 、、如果未指定 encoding，自动识别文件编码类型，避免langchain loader 加载文件报编码错误
            with open(file_path, "rb") as struct_file:
                # 、、自动检测文件的字符编码
                encode_detect = chardet.detect(struct_file.read())
            if encode_detect is None: 
                # 、、检测不出来就默认给个utf-8
                encode_detect = {"encoding": "utf-8"}
            loader_kwargs["encoding"] = encode_detect["encoding"]
    # 、、以下2个加载器都是用langchain_community.document_loader中的，需要传入参数，jq_schema,text_content两个参数
    elif loader_name == "JSONLoader":
        loader_kwargs.setdefault("jq_schema", ".")
        loader_kwargs.setdefault("text_content", False)
    elif loader_name == "JSONLinesLoader":
        loader_kwargs.setdefault("jq_schema", ".")
        loader_kwargs.setdefault("text_content", False)
    # 将文件路径，和一些加载器必要的入参传进加载器，并实例化一个加载器（loader）出来
    loader = DocumentLoader(file_path, **loader_kwargs)
    return loader


@lru_cache()
def make_text_splitter(splitter_name, chunk_size, chunk_overlap):
    """
    根据参数获取特定的分词器
    """
    # 默认使用的是作者实现的中文切割器。
    splitter_name = splitter_name or "SpacyTextSplitter"
    try:
        if (
            splitter_name == "MarkdownHeaderTextSplitter"
        ):  # MarkdownHeaderTextSplitter特殊判定 
            # 、、对markdown的header进行特殊识别，      
            #  "headers_to_split_on": [
            #     ("#", "head1"),
            #     ("##", "head2"),
            #     ("###", "head3"),
            #     ("####", "head4"),
            # ]
            headers_to_split_on = Settings.kb_settings.text_splitter_dict[splitter_name][
                "headers_to_split_on"
            ]
            text_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on, strip_headers=False
            )
        else:
            try:  # 优先使用用户自定义的text_splitter
                text_splitter_module = importlib.import_module("chatchat.server.file_rag.text_splitter")
                TextSplitter = getattr(text_splitter_module, splitter_name)
            except:  # 否则使用langchain的text_splitter
                text_splitter_module = importlib.import_module(
                    "langchain.text_splitter"
                )
                TextSplitter = getattr(text_splitter_module, splitter_name)

            if (
                Settings.kb_settings.text_splitter_dict[splitter_name]["source"] == "tiktoken"
            ):  # 从tiktoken加载（、、tiktoken是OpenAI开发的一个快速BPE分词器，用于将文本转换成模型可以理解的token序列）
                try:
                    text_splitter = TextSplitter.from_tiktoken_encoder(
                        encoding_name=Settings.kb_settings.text_splitter_dict[splitter_name][
                            "tokenizer_name_or_path"
                        ],
                        pipeline="zh_core_web_sm",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                except:
                    text_splitter = TextSplitter.from_tiktoken_encoder(
                        encoding_name=Settings.kb_settings.text_splitter_dict[splitter_name][
                            "tokenizer_name_or_path"
                        ],
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
            elif (
                Settings.kb_settings.text_splitter_dict[splitter_name]["source"] == "huggingface"
            ):  # 从huggingface加载
                if (
                    Settings.kb_settings.text_splitter_dict[splitter_name]["tokenizer_name_or_path"]
                    == "gpt2"
                ):
                    from langchain.text_splitter import CharacterTextSplitter
                    from transformers import GPT2TokenizerFast

                    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                else:  # 字符长度加载
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(
                        Settings.kb_settings.text_splitter_dict[splitter_name]["tokenizer_name_or_path"],
                        trust_remote_code=True,
                    )
                text_splitter = TextSplitter.from_huggingface_tokenizer(
                    tokenizer=tokenizer,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            else:
                try:
                    text_splitter = TextSplitter(
                        pipeline="zh_core_web_sm",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                except:
                    text_splitter = TextSplitter(
                        chunk_size=chunk_size, chunk_overlap=chunk_overlap
                    )
    except Exception as e:
        print(e)
        text_splitter_module = importlib.import_module("langchain.text_splitter")
        TextSplitter = getattr(text_splitter_module, "RecursiveCharacterTextSplitter")
        text_splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # If you use SpacyTextSplitter you can use GPU to do split likes Issue #1287
    # text_splitter._tokenizer.max_length = 37016792
    # text_splitter._tokenizer.prefer_gpu()
    return text_splitter


class KnowledgeFile:
    def __init__(
        self,
        filename: str,
        knowledge_base_name: str,
        loader_kwargs: Dict = {},
    ):
        """
        对应知识库目录中的文件，必须是磁盘上存在的才能进行向量化等操作。
        """
        # 、、 知识库名称
        self.kb_name = knowledge_base_name
        # 、、文件名称
        self.filename = str(Path(filename).as_posix())
        # 、、文件扩展名（如 .txt .html等）
        self.ext = os.path.splitext(filename)[-1].lower()
        if self.ext not in SUPPORTED_EXTS:
            # 、、判断是否支持文件类型
            raise ValueError(f"暂未支持的文件格式 {self.filename}")
        # 、、【沒看到这个参数哪里有传进来过】
        self.loader_kwargs = loader_kwargs
        #、、文件路径（重点注意，初始化的属性是实例被创建的时候赋值过一次的，比如filepath在实例化的时候已经有了具体的路径，后面计算是根据这个来查找的文件，比如执行   实例.file2text() 方法，拿的就是这个path）
        self.filepath = get_file_path(knowledge_base_name, filename)
        # 、、文档
        self.docs = None
        # 、、切割过后的文档
        self.splited_docs = None
        # 、、文件加载器的名字（根据扩展名而来）
        self.document_loader_name = get_LoaderClass(self.ext)
        # 、、文件切割器的名称（默认是作者实现的中文文件切割器）
        self.text_splitter_name = Settings.kb_settings.TEXT_SPLITTER_NAME

    # 、、file：指的是knowledgeFile实例，代表的是磁盘上的"一个文件"。它有属性，例如filepath，ext，documnet_loader_name
    # 、、      splited_docs等，调用file.file2docs会把这个文件  加载成文档  并保存到file.docs上
    # 、、docs: 指的是由加载器（laoder）返回的文档列表，类型通常是List[langchain.docstore.document.Document]
    # 、、      每个Document包含page_content(文本)和metadata(例如soure/页码等)
    def file2docs(self, refresh: bool = False):
        if self.docs is None or refresh:
            logger.info(f"{self.document_loader_name} used for {self.filepath}")
            #、、 document_loader_name (在初始化的时候定义了，是根据文件的扩展名自动计算而来)
            # 、、filepath（初始化的时候定义了，实例化的时候穿进来是filename，但是通过知识库的名称和一些固定规则，如知识库名称下面的content文件夹，来拼接起来的filepath路径）
            #、、 loader_kwargs： 不知道哪里来的，初始化的时候我就没找到哪里穿进来的
            loader = get_loader(
                loader_name=self.document_loader_name,
                file_path=self.filepath,
                loader_kwargs=self.loader_kwargs,
            )
            # 、、TextLoader 类型为 langchain.community_loader的加载器类型
            if isinstance(loader, TextLoader):
                loader.encoding = "utf8"
                self.docs = loader.load()
            else:
                self.docs = loader.load()
        # docs为一个List[Document]，每个Document包括： page_content： 文本内存，字符串类型，metadata： 元数据，字典类型，包含文件名、来源，位置等，
        # Document(
        #     page_content="具体的文本内容",  # 字符串
        #     metadata={                     # 字典，包含各种元数据
        #         'source': '文件路径',
        #         'filename': '文件名',
        #         'filetype': 'text',
        #         'languages': ['zh'],
        #         'category': '文本类型',
        #         'page_number': 1,
        #         # ... 其他元数据
        #     }
        # )
        return self.docs

    def docs2texts(
        self,
        docs: List[Document] = None,
        zh_title_enhance: bool = Settings.kb_settings.ZH_TITLE_ENHANCE,
        refresh: bool = False,
        chunk_size: int = Settings.kb_settings.CHUNK_SIZE,
        chunk_overlap: int = Settings.kb_settings.OVERLAP_SIZE,
        text_splitter: TextSplitter = None,
    ):
        # 这个地方应该只是为了做个保护，如果没有传docs就自己在获取一次，
        docs = docs or self.file2docs(refresh=refresh)
        # 如果是docs为None
        if not docs:
            return []
        # 如果文件的扩展名不是.csv,就进行下面的文本切割环节
        if self.ext not in [".csv"]:
            if text_splitter is None:
                # 切割器为None（即没有设置），获取文本切割器
                text_splitter = make_text_splitter(
                    splitter_name=self.text_splitter_name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )   
            if self.text_splitter_name == "MarkdownHeaderTextSplitter":
                docs = text_splitter.split_text(docs[0].page_content)
            else:
                # 、、使用切割器对生成的docs（List[Document]）的进行切割,并返回一个新的List[Document],Document中的content更加的短，并且加入新的元数据如index，切割前的信息等等
                docs = text_splitter.split_documents(docs)

        if not docs:
            return []

        print(f"文档切分示例：{docs[0]}")
        if zh_title_enhance:
            # 、、如果开启文章标题加强，会额外加入一些提示词，将title的信息加入到page_content中，并且加入额外的元数据
            docs = func_zh_title_enhance(docs)
        self.splited_docs = docs
        # 、、将切割器切割并重新生成的List[Document] 返回，每个Document中包含有page_content，metadata
        return self.splited_docs

    def file2text(
        self,
        zh_title_enhance: bool = Settings.kb_settings.ZH_TITLE_ENHANCE,
        refresh: bool = False,
        chunk_size: int = Settings.kb_settings.CHUNK_SIZE,
        chunk_overlap: int = Settings.kb_settings.OVERLAP_SIZE,
        text_splitter: TextSplitter = None,
    ):
        if self.splited_docs is None or refresh:
            # 返回一个List[Document]
            docs = self.file2docs()
            self.splited_docs = self.docs2texts(
                docs=docs,
                zh_title_enhance=zh_title_enhance,
                refresh=refresh,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                text_splitter=text_splitter,
            )
        # 将切割器切割并重新生成的List[Document] 返回， 每个Document中包含有page_content，metadata
        return self.splited_docs

    def file_exist(self):
        return os.path.isfile(self.filepath)

    def get_mtime(self):
        return os.path.getmtime(self.filepath)

    def get_size(self):
        return os.path.getsize(self.filepath)


# 这个方法主要是对文档进行了切割并且生成page_content 更小的Document，并且进行组装List。
# 返回 第一参数，切割成功状态，第二个是个元组（第一个参数是知识库名称，第二个参数是文件名称，第三个参数是切割过后的Document的page_content更加小的List[Document]
def files2docs_in_thread_file2docs(
    *, file: KnowledgeFile, **kwargs
) -> Tuple[bool, Tuple[str, str, List[Document]]]:
    try:
        # 文件加载
        return True, (file.kb_name, file.filename, file.file2text(**kwargs))
    except Exception as e:
        msg = f"从文件 {file.kb_name}/{file.filename} 加载文档时出错：{e}"
        logger.error(f"{e.__class__.__name__}: {msg}")
        return False, (file.kb_name, file.filename, msg)

# 、、接收的第一个参数files可以是3种类型
# 、、1、KnowledgeFile实例 
#    2、tuple(filename,kb_name) 元组
#    3、dict 包含有filename 和 kb_name （其余健会被当作loader/splitter参数）
def files2docs_in_thread(
    files: List[Union[KnowledgeFile, Tuple[str, str], Dict]],
    chunk_size: int = Settings.kb_settings.CHUNK_SIZE,
    chunk_overlap: int = Settings.kb_settings.OVERLAP_SIZE,
    zh_title_enhance: bool = Settings.kb_settings.ZH_TITLE_ENHANCE,
) -> Generator:
    """
    利用多线程批量将磁盘文件转化成langchain Document.
    如果传入参数是Tuple，形式为(filename, kb_name)
    生成器返回值为 status, (kb_name, file_name, docs | error)
    """

    kwargs_list = []
    for i, file in enumerate(files):
        kwargs = {}
        try:
            # 如果是files元组类型
            if isinstance(file, tuple) and len(file) >= 2:
                # 、、元组的第一项为filename，第二项为kb_name
                filename = file[0]
                kb_name = file[1]
                # 、、通过元组中的filename和kb_name，生成一个Knowledge的实例
                file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
            #、、 如果是字典类型 
            elif isinstance(file, dict):
                filename = file.pop("filename")
                kb_name = file.pop("kb_name")
                # 、、其余键会被当作loader/splitter参数
                kwargs.update(file)
                # 、、通过字典中的filename和kb_name，生成一个Knowledge的实例
                file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)

            kwargs["file"] = file  #、、最终这file还是要以Knowledge实例的形式来承载
            kwargs["chunk_size"] = chunk_size #知识库中单段文本的长度
            kwargs["chunk_overlap"] = chunk_overlap #知识库中相邻文本重合长度
            kwargs["zh_title_enhance"] = zh_title_enhance#是否开启标题加强
            # 将  file (Knowledge实例)，chunk_size, chuck_overlap, zh_title_enhance组成的对象
            # append到kwargs_list中，方便在线程池中取批量进行读取
            kwargs_list.append(kwargs)
        except Exception as e:
            yield False, (kb_name, filename, str(e))

    # 在线程池中批量进行切割组装逻辑，具体来讲就是执行files2docs_in_thread_file2docs方法
    # files2docs_in_thread_file2docs的执行会接收到上面设置的文件实例，单段落文本长度大小等等参数，最终会体现在切割器中的参数（这里修改其实指向的是langchain的类的chunk_sized等，作者的自定义切割器也是继承了langchain的切割器的类）
    # 至此返回一个由yield控制的可迭代对象，方便将文本切割后的List[Document]返回出去
    for result in run_in_thread_pool(
        func=files2docs_in_thread_file2docs, params=kwargs_list
    ):
        yield result


def format_reference(kb_name: str, docs: List[Dict], api_base_url: str="") -> List[Dict]:
    '''
    将知识库检索结果格式化为参考文档的格式
    '''
    from chatchat.server.utils import api_address
    api_base_url = api_base_url or api_address(is_public=True)

    source_documents = []
    for inum, doc in enumerate(docs):
        filename = doc.get("metadata", {}).get("source")
        parameters = urlencode(
            {
                "knowledge_base_name": kb_name,
                "file_name": filename,
            }
        )
        api_base_url = api_base_url.strip(" /")
        url = (
            f"{api_base_url}/knowledge_base/download_doc?" + parameters
        )
        page_content = doc.get("page_content")
        ref = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{page_content}\n\n"""
        source_documents.append(ref)
    
    return source_documents


if __name__ == "__main__":
    from pprint import pprint

    kb_file = KnowledgeFile(
        filename="E:\\LLM\\Data\\Test.md", knowledge_base_name="samples"
    )
    # kb_file.text_splitter_name = "RecursiveCharacterTextSplitter"
    kb_file.text_splitter_name = "MarkdownHeaderTextSplitter"
    docs = kb_file.file2docs()
    # pprint(docs[-1])
    texts = kb_file.docs2texts(docs)
    for text in texts:
        print(text)
