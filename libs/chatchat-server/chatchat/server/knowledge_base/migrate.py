import os
from datetime import datetime
from typing import List, Literal

from dateutil.parser import parse

from chatchat.settings import Settings
from chatchat.server.db.base import Base, engine
from chatchat.server.db.models.conversation_model import ConversationModel
from chatchat.server.db.models.message_model import MessageModel
from chatchat.server.db.repository.knowledge_file_repository import (
    add_file_to_db,
)

# ensure Models are imported
from chatchat.server.db.repository.knowledge_metadata_repository import (
    add_summary_to_db,
)
# ensure Models are imported
from chatchat.server.db.repository.mcp_connection_repository import (
    create_mcp_profile,
)
from chatchat.server.db.session import session_scope
from chatchat.server.knowledge_base.kb_service.base import (
    KBServiceFactory,
    SupportedVSType,
)
from chatchat.server.knowledge_base.utils import (
    KnowledgeFile,
    files2docs_in_thread,
    get_file_path,
    list_files_from_folder,
    list_kbs_from_folder,
)
from chatchat.utils import build_logger
from chatchat.server.utils import get_default_embedding


logger = build_logger()


def create_tables():
    Base.metadata.create_all(bind=engine)
    """
    Base.metadata.create_all(engine)的作用：
        ·在指定的数据库引擎上为Base中注册的所有的ORM模型创建表（基于Base.metadata收集到的Table定义）
        ·只发起CREATE TABLE 语句；如果表已经存在默认会跳过，不会修改已经存在的表的结构
        ·常用于初始化SQLite/测试环境或快速创建表。生成环境的模式迁移应该使用ALembic等迁移工具
    """

def reset_tables():
    Base.metadata.drop_all(bind=engine)
    create_tables()


def import_from_db(
    sqlite_path: str = None,
    # csv_path: str = None,
) -> bool:
    """
    在知识库与向量库无变化的情况下，从备份数据库中导入数据到 info.db。
    适用于版本升级时，info.db 结构变化，但无需重新向量化的情况。
    请确保两边数据库表名一致，需要导入的字段名一致
    当前仅支持 sqlite
    """
    import sqlite3 as sql
    from pprint import pprint

    models = list(Base.registry.mappers)

    try:
        con = sql.connect(sqlite_path)
        con.row_factory = sql.Row
        cur = con.cursor()
        tables = [
            x["name"]
            for x in cur.execute(
                "select name from sqlite_master where type='table'"
            ).fetchall()
        ]
        for model in models:
            table = model.local_table.fullname
            if table not in tables:
                continue
            print(f"processing table: {table}")
            with session_scope() as session:
                for row in cur.execute(f"select * from {table}").fetchall():
                    data = {k: row[k] for k in row.keys() if k in model.columns}
                    if "create_time" in data:
                        data["create_time"] = parse(data["create_time"])
                    pprint(data)
                    session.add(model.class_(**data))
        con.close()
        return True
    except Exception as e:
        print(f"无法读取备份数据库：{sqlite_path}。错误信息：{e}")
        return False


def file_to_kbfile(kb_name: str, files: List[str]) -> List[KnowledgeFile]:
    """
    list实例中的每个实例中主要提供包含有 file2docs，file2text, docs2text,这三个转化方法，
    和一些如文件路径filepath，splited_docs等的关键属性
    """
    kb_files = []
    for file in files:
        try:
            # 、、返回一个知识库文件实例，（2个入参，第一个是要向量化的文件，第二个是知识库名称）
            # 、、这个实例中主要提供包含有 file2docs，file2text, docs2text,这三个转化方法，和一些如splited_docs等的关键属性
            kb_file = KnowledgeFile(filename=file, knowledge_base_name=kb_name)
            kb_files.append(kb_file)
        except Exception as e:
            msg = f"{e}，已跳过"
            logger.error(f"{e.__class__.__name__}: {msg}")
    return kb_files


def folder2db(
    kb_names: List[str],
    mode: Literal["recreate_vs", "update_in_db", "increment"],
    vs_type: Literal["faiss", "milvus", "pg", "chromadb"] = Settings.kb_settings.DEFAULT_VS_TYPE,
    embed_model: str = get_default_embedding(),
    chunk_size: int = Settings.kb_settings.CHUNK_SIZE,
    chunk_overlap: int = Settings.kb_settings.OVERLAP_SIZE,
    zh_title_enhance: bool = Settings.kb_settings.ZH_TITLE_ENHANCE,
):
    """
    指定的本地知识库文件夹内容导入/重建到向量检索数据库(vector store)中
        入参说明：
            kb_names: 知识库名称list
            mode:
                recreate_vs(清除向量库，从本地文件重建)
                update_in_db(以数据库中文件列表为基准，利用本地文件更新向量库)
                increment(对比本地目录与数据库中的文件列表，进行增量向量化)
            vs_type: (vector store)向量存储/检索 类型
    """
    
    def files2vs(kb_name: str, kb_files: List[KnowledgeFile]) -> List:
        result = []
        for success, res in files2docs_in_thread(
            kb_files, # 传入知识库文件实例list,实例中包含由file2docs,docs2text,file2text方法和一些关键如filepath等关键属性
            chunk_size=chunk_size,#知识库中单段文本长度
            chunk_overlap=chunk_overlap,  #知识库中相邻文本重合长度
            zh_title_enhance=zh_title_enhance,  #是否开启中文标题加强，以及标题增强的相关配置
        ):
            if success:
                _, filename, docs = res
                print(
                    f"正在将 {kb_name}/{filename} 添加到向量库，共包含{len(docs)}条文档"
                )
                kb_file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
                kb_file.splited_docs = docs
                kb.add_doc(kb_file=kb_file, not_refresh_vs_cache=True)
                result.append({"kb_name": kb_name, "file": filename, "docs": docs})
            else:
                print(res)
        return result

    kb_names = kb_names or list_kbs_from_folder()
    """、、获取知识库的文件夹list，作为知识库的名称list"""
    for kb_name in kb_names:
        start = datetime.now()
        # 、、通过知识库工厂：入参为 知识库名称，向量库类型，嵌入模型名称   
        # 、、返回一个知识库的实例，里面包含了对知识库的各种方法，每种不同的vs_type,会返回不同的方法实现
        kb = KBServiceFactory.get_service(kb_name, vs_type, embed_model)
        if not kb.exists():
            kb.create_kb()

        # 清除向量库，从本地文件重建
        if mode == "recreate_vs":
            # 、、清空这个向量知识库表，主要是执行了knowlange_file和file_doc表中kb_name为当前kb_bamed的所有行
            kb.clear_vs()
            # 、、新建一个向量知识库，操作包括创建知识库content文件夹，实例化一个知识库ORM模型，并添加到事务中
            kb.create_kb() 
            # 、、普通文件转化为知识库文件，2个入参，第一个是知识库名称（文件夹名称），第二个是知识库（文件夹）下的所有文件list，
            # 、、返回存储向量知识库需要的文件实例list，
            kb_files = file_to_kbfile(kb_name, list_files_from_folder(kb_name))
            # 、、将从知识库文件实例list转化存储到向量库
            result = files2vs(kb_name, kb_files)
            kb.save_vector_store()
        # # 不做文件内容的向量化，仅将文件元信息存到数据库
        # # 由于现在数据库存了很多与文本切分相关的信息，单纯存储文件信息意义不大，该功能取消。
        # elif mode == "fill_info_only":
        #     files = list_files_from_folder(kb_name)
        #     kb_files = file_to_kbfile(kb_name, files)
        #     for kb_file in kb_files:
        #         add_file_to_db(kb_file)
        #         print(f"已将 {kb_name}/{kb_file.filename} 添加到数据库")
        # 以数据库中文件列表为基准，利用本地文件更新向量库
        elif mode == "update_in_db":
            files = kb.list_files()
            kb_files = file_to_kbfile(kb_name, files)
            result = files2vs(kb_name, kb_files)
            kb.save_vector_store()
        # 对比本地目录与数据库中的文件列表，进行增量向量化
        elif mode == "increment":
            db_files = kb.list_files()
            folder_files = list_files_from_folder(kb_name)
            files = list(set(folder_files) - set(db_files))
            kb_files = file_to_kbfile(kb_name, files)
            result = files2vs(kb_name, kb_files)
            kb.save_vector_store()
        else:
            print(f"unsupported migrate mode: {mode}")
        end = datetime.now()
        kb_path = (
            f"知识库路径\t：{kb.kb_path}\n"
            if kb.vs_type() == SupportedVSType.FAISS
            else ""
        )
        file_count = len(kb_files)
        success_count = len(result)
        docs_count = sum([len(x["docs"]) for x in result])
        print("\n" + "-" * 100)
        print(
            (
                f"知识库名称\t：{kb_name}\n"
                f"知识库类型\t：{kb.vs_type()}\n"
                f"向量模型：\t：{kb.embed_model}\n"
            )
            + kb_path
            + (
                f"文件总数量\t：{file_count}\n"
                f"入库文件数\t：{success_count}\n"
                f"知识条目数\t：{docs_count}\n"
                f"用时\t\t：{end-start}"
            )
        )
        print("-" * 100 + "\n")


def prune_db_docs(kb_names: List[str]):
    """
    delete docs in database that not existed in local folder.
    it is used to delete database docs after user deleted some doc files in file browser
    """
    for kb_name in kb_names:
        kb = KBServiceFactory.get_service_by_name(kb_name)
        if kb is not None:
            files_in_db = kb.list_files()
            files_in_folder = list_files_from_folder(kb_name)
            files = list(set(files_in_db) - set(files_in_folder))
            kb_files = file_to_kbfile(kb_name, files)
            for kb_file in kb_files:
                kb.delete_doc(kb_file, not_refresh_vs_cache=True)
                print(f"success to delete docs for file: {kb_name}/{kb_file.filename}")
            kb.save_vector_store()


def prune_folder_files(kb_names: List[str]):
    """
    delete doc files in local folder that not existed in database.
    it is used to free local disk space by delete unused doc files.
    """
    for kb_name in kb_names:
        kb = KBServiceFactory.get_service_by_name(kb_name)
        if kb is not None:
            files_in_db = kb.list_files()
            files_in_folder = list_files_from_folder(kb_name)
            files = list(set(files_in_folder) - set(files_in_db))
            for file in files:
                os.remove(get_file_path(kb_name, file))
                print(f"success to delete file: {kb_name}/{file}")
