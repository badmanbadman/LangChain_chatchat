import json

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import DeclarativeMeta, declarative_base
from sqlalchemy.orm import sessionmaker

from chatchat.settings import Settings
"""
SQLAlchemy 是一个用于在Python中于和 关系型数据库   进行交互的库，主要提供2层功能：
    Core (SQL Expression Language): 以Python方式构建和执行原生SQL查询，适合需要精细化控制SQL的场景
    ORM (对象关系映射): 将数据库表映射为Python类，能用面向对象的方法进行增删查改数据
主要概念
    Engine: 数据库连接工厂（create_engine）
    MetaData/Table: 表结构的元数据表示
    Declarative Base: 通过declarative_base() 定义ORM模型类的基类。
    Session: 管理对象于数据库会话，事务（通过sessionmaker创建）
    支持多种数据库（SQLite, Postgres, MySQL等）并常与Alembic(数据库迁移工具)配合做迁移（migrations）
"""

engine = create_engine(
    Settings.basic_settings.SQLALCHEMY_DATABASE_URI,
    json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
)
"""
engine: 数据库引擎，
url: 数据库连接 URL（必需）。格式示例：
    SQLite 文件: "sqlite:///./data.db"
    Postgres: "postgresql+psycopg2://user:pass@host:port/dbname"
echo: bool，是否把执行的 SQL 打印到 stdout（调试用）。相当于 SQL 日志级别。
future: bool，启用 SQLAlchemy 2.0 风格行为（在 1.4 中用于过渡）。
pool_size: 连接池的固定大小（适用于某些 DBAPI 的连接池实现）。
max_overflow: 超出 pool_size 后允许额外创建的连接数（临时连接）。
pool_timeout: 从连接池获取连接的超时时间（秒）。
pool_recycle: 自动重置连接的秒数（避免 DB 端断开超时）。
poolclass: 指定自定义连接池类（如 NullPool、QueuePool 等）。
connect_args: dict，传递给底层 DBAPI.connect 的额外参数（例如 sqlite 的 timeout、check_same_thread）。
isolation_level: 事务隔离级别（如 "READ COMMITTED" 等）。
json_serializer: 自定义 JSON 序列化函数（用于支持 JSON 类型字段或在某些方言/扩展写入 JSON），你代码里用来确保 json 序列化时不转义中文：
lambda obj: json.dumps(obj, ensure_ascii=False)
encoding / convert_unicode（较旧参数，现代 SQLAlchemy 多使用 connect_args 或 DBAPI 配置）。
creator: 提供一个自定义创建底层连接的可调用（高级用法）。
"""

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
"""
sessionmaker(...)返回一个Session工厂类，（不是具体的连接），调用Sessionlocal() 会得到一个新的Session实例，
用于数据库会话和事务管理
参数说明：
autocommit=False: 不会自动提交事务，需手动嗲用session.commit() 
autoflush=False: 在执行查询前不会自动把未提交的变更 flush (刷新) 到数据库（避免自动同步，如有需要可以显示的session.flush()）
bind=engine: 把这个SessionFactory绑定到上面创建的engine，使得生成的Session使用该数据库连接引擎。
    常见用法示例：
    db = SessionLocal()
    try:
        obj = db.query(MyModel).filter_by(id=1).first()
        obj.name = 'new'
        db.commit()
    except:
        db.rollback()
        raise
    finally:
        db.close()
"""
Base: DeclarativeMeta = declarative_base()
""" 、、
declarative_base()：创建并返回一个基类（Base），所有的ORM模型都应该继承自它，
这个Base包含metadata（表/映射信息）
ORM 模型： （Object-Relational Mapping）是把关系型数据库和面向对象语言的类/对象进行映射的技术
ORM模型就是映射类：类->表，实例->表中的一行，属性->列。
通过ORM可以用面向对象的方式增删查改数据库，而不是直接写原始的SQL（但是仍可以执行SQL）
"""
