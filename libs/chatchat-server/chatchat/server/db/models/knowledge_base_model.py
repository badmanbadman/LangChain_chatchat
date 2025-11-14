from datetime import datetime
from typing import Optional

from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Integer, String, func

from chatchat.server.db.base import Base

# 目的：
#   ORM模型（SQLAlchemy的类）： 用于定义数据库表结构，映射和持久化（读写数据库）
#   Pydantic模型： 用于数据验证、序列化、反序列化，API请求、响应或内部DTO
# 是否持久化：ORM模型负责持久化到DB，Pydantic不直接与DB交互
# 行为：ORM由Column，关系，Session等，Pydantic主要做校验，到处JSON等
# 互转： 常把ORM实例转换成Pydantic schema（用于返回API响应，）需要开启orm/from_attributes支持
class KnowledgeBaseModel(Base):
    """
    知识库模型
    、、这是一个SQLAlchemy ORM模型，映射到数据库表 knowledge_base ,字段含义
        id: 整数主键，自增
        kb_name: 知识库名称
        kb_info: 知识库简介
        vs_type: 向量库类型（faiss，milvus等）
        embedd_model: 嵌入embedding模型的名称
        file_count: 文件数量（默认0）
        create_time: 创建时间，默认使用数据库函数 now()
    """

    __tablename__ = "knowledge_base"
    id = Column(Integer, primary_key=True, autoincrement=True, comment="知识库ID")
    kb_name = Column(String(50), comment="知识库名称")
    kb_info = Column(String(200), comment="知识库简介(用于Agent)")
    vs_type = Column(String(50), comment="向量库类型")
    embed_model = Column(String(50), comment="嵌入模型名称")
    file_count = Column(Integer, default=0, comment="文件数量")
    create_time = Column(DateTime, default=func.now(), comment="创建时间")

    # 用于调试/日志输出模型实例可读表示
    def __repr__(self):
        return (
            f"<KnowledgeBase(id='{self.id}', kb_name='{self.kb_name}', "
            f"kb_intro='{self.kb_info}', vs_type='{self.vs_type}', "
            f"embed_model='{self.embed_model}', file_count='{self.file_count}', "
            f"create_time='{self.create_time}')>"
        )

# 创建一个对应的 Pydantic 模型
class KnowledgeBaseSchema(BaseModel):
    id: int
    kb_name: str
    kb_info: Optional[str]
    vs_type: Optional[str]
    embed_model: Optional[str]
    file_count: Optional[int]
    create_time: Optional[datetime]

    class Config:
        from_attributes = True  # 确保可以从 ORM 实例进行验证
