from chatchat.server.db.models.knowledge_base_model import (
    KnowledgeBaseModel,
    KnowledgeBaseSchema,
)
from chatchat.server.db.session import with_session


@with_session
def add_kb_to_db(session, kb_name, kb_info, vs_type, embed_model):
    # 、、session是通过with_session注解来的，后面的参数是调用add_kb_to_db来的

    
    # 查找 关系型数据库中 knowledge_base 这表里面， kb_name 为传进来的 kb_name的
    # 再ROM的表中一行所映射的是一个知识库实例，如下面查找到的kb其实就映射为名字为传进来的kb_name的实例
    kb = (
        session.query(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.kb_name.ilike(kb_name))
        .first()
    )

    # 没找到知识库实例就添加一个
    if not kb:
        # 实例化一个知识库ORM模型,里面包含了kb_name,kb_info,vs_type,embed_model（这里存的是嵌入模型的名字，不是模型哦，要区分下的）
        kb = KnowledgeBaseModel(
            kb_name=kb_name, kb_info=kb_info, vs_type=vs_type, embed_model=embed_model
        )
        # 添加到事务中
        session.add(kb)
    else:  # update kb with new vs_type and embed_model
        # 找到了知识库实例，更新知识库实例的字段  kb_info（ 知识库简介）vs_type（向量库类型，如fiass等）  embed_model
        # 注意下面并没有显示的调用更新数据库的逻辑，因为我们的使用的是SQLAIchemy它会自动变更跟踪机制
        # 这个kb 是通过session.query来获取的，这个kb对象以及被SQLAIchemy Session跟踪管理，
        # Session 维护了一个Identity Map，记录了所有从数据库加载的对象
        # 对这些对象的任何修改都会被自动检测
        # 当修改已加载的对象的属性时候,SQLAIchemy会检测到属性值的变化,将该对象标记为脏,在事务提交的时候自动生成UPDATE语句
        kb.kb_info = kb_info
        kb.vs_type = vs_type
        kb.embed_model = embed_model
    # 返回一个布尔值，
    return True


@with_session
def list_kbs_from_db(session, min_file_count: int = -1):
    kbs = (
        session.query(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.file_count > min_file_count)
        .all()
    )
    kbs = [KnowledgeBaseSchema.model_validate(kb) for kb in kbs]
    return kbs


@with_session
def kb_exists(session, kb_name):
    kb = (
        session.query(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.kb_name.ilike(kb_name))
        .first()
    )
    status = True if kb else False
    return status


@with_session
def load_kb_from_db(session, kb_name):
    kb = (
        session.query(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.kb_name.ilike(kb_name))
        .first()
    )
    if kb:
        kb_name, vs_type, embed_model = kb.kb_name, kb.vs_type, kb.embed_model
    else:
        kb_name, vs_type, embed_model = None, None, None
    return kb_name, vs_type, embed_model


@with_session
def delete_kb_from_db(session, kb_name):
    kb = (
        session.query(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.kb_name.ilike(kb_name))
        .first()
    )
    if kb:
        session.delete(kb)
    return True


@with_session
def get_kb_detail(session, kb_name: str) -> dict:
    kb: KnowledgeBaseModel = (
        session.query(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.kb_name.ilike(kb_name))
        .first()
    )
    if kb:
        return {
            "kb_name": kb.kb_name,
            "kb_info": kb.kb_info,
            "vs_type": kb.vs_type,
            "embed_model": kb.embed_model,
            "file_count": kb.file_count,
            "create_time": kb.create_time,
        }
    else:
        return {}
