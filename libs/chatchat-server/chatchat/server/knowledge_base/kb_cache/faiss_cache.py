import os

from langchain.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain.vectorstores.faiss import FAISS

from chatchat.settings import Settings
from chatchat.server.knowledge_base.kb_cache.base import * # 注意这里：引入了base.py中的所有 CachePool, ThreadSafeObject, logger
from chatchat.server.knowledge_base.utils import get_vs_path
from chatchat.server.utils import get_Embeddings, get_default_embedding


# patch FAISS to include doc id in Document.metadata
def _new_ds_search(self, search: str) -> Union[str, Document]:
    if search not in self._dict:
        return f"ID {search} not found."
    else:
        doc = self._dict[search]
        if isinstance(doc, Document):
            doc.metadata["id"] = search
        return doc


InMemoryDocstore.search = _new_ds_search


# 这里继承了ThreadSafeObject后，又
# 对Faiss的特定操作进行了扩展，添加了save，clear，docs_count的操作，
# 对于调试日志也加入了docs_count
class ThreadSafeFaiss(ThreadSafeObject):
    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"<{cls}: key: {self.key}, obj: {self._obj}, docs_count: {self.docs_count()}>"

    def docs_count(self) -> int:
        return len(self._obj.docstore._dict)

    def save(self, path: str, create_path: bool = True):
        with self.acquire():
            if not os.path.isdir(path) and create_path:
                os.makedirs(path)
            ret = self._obj.save_local(path)
            logger.info(f"已将向量库 {self.key} 保存到磁盘")
        return ret

    def clear(self):
        ret = []
        with self.acquire():
            ids = list(self._obj.docstore._dict.keys())
            if ids:
                ret = self._obj.delete(ids)
                assert len(self._obj.docstore._dict) == 0
            logger.info(f"已将向量库 {self.key} 清空")
        return ret


class _FaissPool(CachePool):
    def new_vector_store(
        self,
        kb_name: str,
        embed_model: str = get_default_embedding(),
    ) -> FAISS:
        """创建一个新的向量库"""
        # 获取嵌入模型
        embeddings = get_Embeddings(embed_model=embed_model)
        doc = Document(page_content="init", metadata={})
        vector_store = FAISS.from_documents([doc], embeddings, normalize_L2=True)
        ids = list(vector_store.docstore._dict.keys())
        # 清空向量库 
        vector_store.delete(ids)
        return vector_store

    def new_temp_vector_store(
        self,
        embed_model: str = get_default_embedding(),
    ) -> FAISS:
        # create an empty vector store
        embeddings = get_Embeddings(embed_model=embed_model)
        doc = Document(page_content="init", metadata={})
        vector_store = FAISS.from_documents([doc], embeddings, normalize_L2=True)
        ids = list(vector_store.docstore._dict.keys())
        vector_store.delete(ids)
        return vector_store

    def save_vector_store(self, kb_name: str, path: str = None):
        if cache := self.get(kb_name):
            return cache.save(path)

    def unload_vector_store(self, kb_name: str):
        if cache := self.get(kb_name):
            self.pop(kb_name)
            logger.info(f"成功释放向量库：{kb_name}")


class KBFaissPool(_FaissPool):
    """
    Faiss向量库的缓存池
    """
    def load_vector_store(
        self,
        kb_name: str,
        vector_name: str = None,
        create: bool = True,
        embed_model: str = get_default_embedding(),
    ) -> ThreadSafeFaiss:
        # 、、获取锁，进行原子操作，
        self.atomic.acquire()
        locked = True
        # 、、向量库名称
        vector_name = vector_name or embed_model.replace(":", "_")
        key = (kb_name, vector_name) # 用元组比拼接字符串好一些 ,(这里的元组是用来当作线程池的key的)
        cache = self.get(key)  
        try:
            # 、、首次进来是None直接走下面的if
            if cache is None:
                # 、、初始化一个线程安全的Faiss向量，将self，即实例化的缓存池放到线程安全类里面去管理，
                # 、、然后将这个线程安全类当作value，存储在实例化的缓存池的有序字典上来存储
                # 这里需要好好理解下: 在ThreadSafeFaiss() 实例里面存储进去了key,和pool,
                # 这个pool是什么?是self,self是KBFaissPool()这个实例,KBFaissPool是继承自_FaissPool,_FaissPool是继承自CachePool,
                # 所以这个self中其实是包含了这三个类中的属性与方法,这个key和pool放进了ThreadSafeFaiss实例对象中并成为了ThreadSafeFaiss实例这对象中的两个属性_key,和_pool,
                # 然后又将这个ThreadSafeFaiss实例,通过self的也就是KBFaissPool()实例的set方法,添加挂载到了KBFaissPool()实例的OrderDict中,缓存了起来,这样下次触发进来这里的时候,就不会再进catch 这里面来了
                # 后续用的时候其实用的是这个线程安全类ThreadSafeFaiss中的所有,比如可以从acquire,save等等
                item = ThreadSafeFaiss(key, pool=self)
                # 、、设置缓存，将这个由知识库名称和向量库名称构成的元组当作 CachePool中的有序字典的缓存键key，将用线程安全类存储的实例当作value，存储在实例化的缓存池的有序字典上
                self.set(key, item)  
                # 这个 item 就等价于 self.get(key)
                with item.acquire(msg="初始化"):
                    self.atomic.release()# 释放锁,因为上面的acquire中有对self的_cache进行操作,所以在这里才释放锁,不得不说,作者很细节了
                    locked = False
                    logger.info(
                        f"loading vector store in '{kb_name}/vector_store/{vector_name}' from disk."
                    )
                    # 根据知识库名称和向量库名称来生成向量库的路径
                    vs_path = get_vs_path(kb_name, vector_name)
                    # 判断index.faiss是否在vs_path路径下。index.faiss是最关键的主要的向量索引文件
                    if os.path.isfile(os.path.join(vs_path, "index.faiss")):
                        # 存在索引文件，就从磁盘加载现有向量库
                        # 获取嵌入模型
                        embeddings = get_Embeddings(embed_model=embed_model)
                        vector_store = FAISS.load_local(
                            vs_path, #保存Faiss索引的目录路径
                            embeddings, #用于初始化Faiss索引的嵌入模型，这个模型应该是和创建索引时用的模型一样
                            normalize_L2=True, #是否在加载索引后对向量进行L2归一化，默认是True，通常是为了保证向量之间的相似度计算（如余弦相似度）正确
                            allow_dangerous_deserialization=True, # 是否允许潜在的不可信反序列化，由于Faiss索引是从磁盘加载的，可能存在恶意索引文件，所以要显式设置位True确认风险
                        )
                    elif create:
                        # 如果不存在，创建一个新的空的向量库

                        # 创建目录
                        if not os.path.exists(vs_path):
                            os.makedirs(vs_path)
                        # 创建向量库 
                        vector_store = self.new_vector_store(
                            kb_name=kb_name, embed_model=embed_model
                        )
                        # 向量库保存到磁盘
                        vector_store.save_local(vs_path)
                    else:
                        raise RuntimeError(f"knowledge base {kb_name} not exist.")
                    # 将向量库保存到安全线程的obj对象上，方便后面获取
                    item.obj = vector_store
                    # 加载完成，唤醒其他所有的等待线程
                    item.finish_loading()
            else:
                # 如缓存中有值，直接释放锁
                self.atomic.release()
                locked = False
        except Exception as e:
            if locked:  # we don't know exception raised before or after atomic.release
                # 报错了就直接释放锁，不要阻塞
                self.atomic.release()
            logger.exception(e)
            raise RuntimeError(f"向量库 {kb_name} 加载失败。")
        # 将向量库返回（先放到了缓存中，所以此处是从缓存中获取）
        return self.get((kb_name, vector_name))


class MemoFaissPool(_FaissPool):
    r"""
    临时向量库的缓存池
    """

    def load_vector_store(
        self,
        kb_name: str,
        embed_model: str = get_default_embedding(),
    ) -> ThreadSafeFaiss:
        self.atomic.acquire()
        cache = self.get(kb_name)
        if cache is None:
            item = ThreadSafeFaiss(kb_name, pool=self)
            self.set(kb_name, item)
            with item.acquire(msg="初始化"):
                self.atomic.release()
                logger.info(f"loading vector store in '{kb_name}' to memory.")
                # create an empty vector store
                vector_store = self.new_temp_vector_store(embed_model=embed_model)
                item.obj = vector_store
                item.finish_loading()
        else:
            self.atomic.release()
        return self.get(kb_name)


kb_faiss_pool = KBFaissPool(cache_num=Settings.kb_settings.CACHED_VS_NUM)
memo_faiss_pool = MemoFaissPool(cache_num=Settings.kb_settings.CACHED_MEMO_VS_NUM)
#
#
# if __name__ == "__main__":
#     import time, random
#     from pprint import pprint
#
#     kb_names = ["vs1", "vs2", "vs3"]
#     # for name in kb_names:
#     #     memo_faiss_pool.load_vector_store(name)
#
#     def worker(vs_name: str, name: str):
#         vs_name = "samples"
#         time.sleep(random.randint(1, 5))
#         embeddings = load_local_embeddings()
#         r = random.randint(1, 3)
#
#         with kb_faiss_pool.load_vector_store(vs_name).acquire(name) as vs:
#             if r == 1: # add docs
#                 ids = vs.add_texts([f"text added by {name}"], embeddings=embeddings)
#                 pprint(ids)
#             elif r == 2: # search docs
#                 docs = vs.similarity_search_with_score(f"{name}", k=3, score_threshold=1.0)
#                 pprint(docs)
#         if r == 3: # delete docs
#             logger.warning(f"清除 {vs_name} by {name}")
#             kb_faiss_pool.get(vs_name).clear()
#
#     threads = []
#     for n in range(1, 30):
#         t = threading.Thread(target=worker,
#                              kwargs={"vs_name": random.choice(kb_names), "name": f"worker {n}"},
#                              daemon=True)
#         t.start()
#         threads.append(t)
#
#     for t in threads:
#         t.join()
