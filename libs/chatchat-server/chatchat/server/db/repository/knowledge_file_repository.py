from typing import Dict, List

from chatchat.server.db.models.knowledge_base_model import KnowledgeBaseModel
from chatchat.server.db.models.knowledge_file_model import (
    FileDocModel,
    KnowledgeFileModel,
)
from chatchat.server.db.session import with_session
from chatchat.server.knowledge_base.utils import KnowledgeFile


@with_session
def list_file_num_docs_id_by_kb_name_and_file_name(
    session,
    kb_name: str,
    file_name: str,
) -> List[int]:
    """
    列出某知识库某文件对应的所有Document的id。
    返回形式：[str, ...]
    """
    doc_ids = (
        session.query(FileDocModel.doc_id)
        .filter_by(kb_name=kb_name, file_name=file_name)
        .all()
    )
    return [int(_id[0]) for _id in doc_ids]


@with_session
def list_docs_from_db(
    session,
    kb_name: str,
    file_name: str = None,
    metadata: Dict = {},
) -> List[Dict]:
    """
    列出某知识库某文件对应的所有Document。
    返回形式：[{"id": str, "metadata": dict}, ...]
    """
    docs = session.query(FileDocModel).filter(FileDocModel.kb_name.ilike(kb_name))
    if file_name:
        docs = docs.filter(FileDocModel.file_name.ilike(file_name))
    for k, v in metadata.items():
        docs = docs.filter(FileDocModel.meta_data[k].as_string() == str(v))

    return [{"id": x.doc_id, "metadata": x.metadata} for x in docs.all()]


@with_session
def delete_docs_from_db(
    session,
    kb_name: str,
    file_name: str = None,
) -> List[Dict]:
    """
    删除某知识库某文件对应的所有Document，并返回被删除的Document。
    返回形式：[{"id": str, "metadata": dict}, ...]
    """
    docs = list_docs_from_db(kb_name=kb_name, file_name=file_name)
    query = session.query(FileDocModel).filter(FileDocModel.kb_name.ilike(kb_name))
    if file_name:
        query = query.filter(FileDocModel.file_name.ilike(file_name))
    query.delete(synchronize_session=False)
    session.commit()
    return docs


@with_session
def add_docs_to_db(session, kb_name: str, file_name: str, doc_infos: List[Dict]):
    """
    将某知识库某文件对应的所有Document信息添加到数据库。
    doc_infos形式：[{"id": str, "metadata": dict}, ...]
    """
    # ! 这里会出现doc_infos为None的情况，需要进一步排查
    if doc_infos is None:
        print(
            "输入的server.db.repository.knowledge_file_repository.add_docs_to_db的doc_infos参数为None"
        )
        return False
    for d in doc_infos:
        obj = FileDocModel(
            kb_name=kb_name,
            file_name=file_name,
            doc_id=d["id"],
            meta_data=d["metadata"],
        )
        # 会话中存储新对象obj，session.commit()的时候执行数据库插入逻辑将数据插入到file_doc表中
        session.add(obj)
    return True


@with_session
def count_files_from_db(session, kb_name: str) -> int:
    return (
        session.query(KnowledgeFileModel)
        .filter(KnowledgeFileModel.kb_name.ilike(kb_name))
        .count()
    )


@with_session
def list_files_from_db(session, kb_name):
    files = (
        session.query(KnowledgeFileModel)
        .filter(KnowledgeFileModel.kb_name.ilike(kb_name))
        .all()
    )
    docs = [f.file_name for f in files]
    return docs


@with_session
def add_file_to_db(
    session,
    kb_file: KnowledgeFile,
    docs_count: int = 0,
    custom_docs: bool = False,
    doc_infos: List[Dict] = [],  # 形式：[{"id": str, "metadata": dict}, ...]
):
    """
    这里是将知识库的信息进行结构化管理，存储的是文件元数据，版本信息，统计信息等主要作用是文件管理，审计，统计分析等待，前面的to_add_doc是将文档向量，原始内容存储到Faiss之类的向量库，用于相似度搜索等
    """
    # 、、在事务中查找到和传进来的kb_name(知识库名称)一样的ORM模型，这kbORM是用来定义表结构，映射和持久化的
    kb = session.query(KnowledgeBaseModel).filter_by(kb_name=kb_file.kb_name).first()
    # 、、这个kb应该一定是有的，在前面的代码逻辑中有个创建的逻辑，这里加个if可能还是为了更加健壮，加个保护
    if kb:
        # 如果已经存在该文件，则更新文件信息与版本号
        existing_file: KnowledgeFileModel = (
            session.query(KnowledgeFileModel)
            .filter(
                KnowledgeFileModel.kb_name.ilike(kb_file.kb_name),
                KnowledgeFileModel.file_name.ilike(kb_file.filename),
            )
            .first()
        ) # 根据知识库名称和文件名称在 knowledge_file 表中查找是否存在（对应ROM为KnowledgeFileModel）
        mtime = kb_file.get_mtime()
        size = kb_file.get_size()

        if existing_file:
            existing_file.file_mtime = mtime
            existing_file.file_size = size
            existing_file.docs_count = docs_count
            existing_file.custom_docs = custom_docs
            existing_file.file_version += 1
        # 否则，添加新文件
        else:
            new_file = KnowledgeFileModel(
                file_name=kb_file.filename,
                file_ext=kb_file.ext,
                kb_name=kb_file.kb_name,
                document_loader_name=kb_file.document_loader_name,
                text_splitter_name=kb_file.text_splitter_name or "SpacyTextSplitter",
                file_mtime=mtime,
                file_size=size,
                docs_count=docs_count,
                custom_docs=custom_docs,
            )
            kb.file_count += 1
            # 将新对象new_file添加到session会话中，准备插入数据库。在session.commit()的时候才真正的将在KnowledgeFileModel 所映射的knowledge_file表中新加一条数据，
            session.add(new_file)
        # 和上面的add_file_to_db一样这里的add_docs_to_db是将文件对应的Documents信息添加到数据库表名为  file_doc
        add_docs_to_db(
            kb_name=kb_file.kb_name, file_name=kb_file.filename, doc_infos=doc_infos
        )
    return True


@with_session
def delete_file_from_db(session, kb_file: KnowledgeFile):
    # 从关系型数据库删除文件信息 
    existing_file = (
        session.query(KnowledgeFileModel)
        .filter(
            KnowledgeFileModel.file_name.ilike(kb_file.filename),
            KnowledgeFileModel.kb_name.ilike(kb_file.kb_name),
        )
        .first()
    )
    if existing_file:
        session.delete(existing_file)
        delete_docs_from_db(kb_name=kb_file.kb_name, file_name=kb_file.filename)
        session.commit()

        kb = (
            session.query(KnowledgeBaseModel)
            .filter(KnowledgeBaseModel.kb_name.ilike(kb_file.kb_name))
            .first()
        )
        if kb:
            kb.file_count -= 1
            session.commit()
    return True


@with_session
def delete_files_from_db(session, knowledge_base_name: str):
    # 、、删除knowledge_file表中，(knowledge_base_name)与kb_name不区分大小写）匹配的所有行
    session.query(KnowledgeFileModel).filter(
        KnowledgeFileModel.kb_name.ilike(knowledge_base_name)
    ).delete(synchronize_session=False)
    # 、、 synchronize_session=False 是直接在数据库层执行批量删除，性能好，但是不会在当前Session的内存对象上同步删除状态（可能导致session中残留已删除对象的缓存），Synchronize_sessionn=False表示不尝试同步

    # 、、删除file_doc表中,(knowledge_base_name)与kb_name（不区分大小写）匹配的所有行
    session.query(FileDocModel).filter(
        FileDocModel.kb_name.ilike(knowledge_base_name)
    ).delete(synchronize_session=False)
    # 、、 knowledge_base表中该知识库的file_count置为0（如果knowledge_base表存在的话）
    kb = (
        session.query(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.kb_name.ilike(knowledge_base_name))
        .first()
    )
    if kb:
        # 将(knowledge_base_name)与kb_name（不区分大小写）匹配的行的file_count 设置为0
        kb.file_count = 0

    session.commit()
    return True


@with_session
def file_exists_in_db(session, kb_file: KnowledgeFile):
    existing_file = (
        session.query(KnowledgeFileModel)
        .filter(
            KnowledgeFileModel.file_name.ilike(kb_file.filename),
            KnowledgeFileModel.kb_name.ilike(kb_file.kb_name),
        )
        .first()
    )
    return True if existing_file else False


@with_session
def get_file_detail(session, kb_name: str, filename: str) -> dict:
    file: KnowledgeFileModel = (
        session.query(KnowledgeFileModel)
        .filter(
            KnowledgeFileModel.file_name.ilike(filename),
            KnowledgeFileModel.kb_name.ilike(kb_name),
        )
        .first()
    )
    if file:
        return {
            "kb_name": file.kb_name,
            "file_name": file.file_name,
            "file_ext": file.file_ext,
            "file_version": file.file_version,
            "document_loader": file.document_loader_name,
            "text_splitter": file.text_splitter_name,
            "create_time": file.create_time,
            "file_mtime": file.file_mtime,
            "file_size": file.file_size,
            "custom_docs": file.custom_docs,
            "docs_count": file.docs_count,
        }
    else:
        return {}
