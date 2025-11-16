import os
import shutil
from typing import Dict, List, Tuple

from langchain.docstore.document import Document

from chatchat.settings import Settings
from chatchat.server.file_rag.utils import get_Retriever
from chatchat.server.knowledge_base.kb_cache.faiss_cache import (
    ThreadSafeFaiss,
    kb_faiss_pool,
)
from chatchat.server.knowledge_base.kb_service.base import KBService, SupportedVSType
from chatchat.server.knowledge_base.utils import KnowledgeFile, get_kb_path, get_vs_path
from langchain.vectorstores.faiss import FAISS


class FaissKBService(KBService):
    vs_path: str
    kb_path: str
    vector_name: str = None

    def vs_type(self) -> str:
        return SupportedVSType.FAISS

    def get_vs_path(self):
        return get_vs_path(self.kb_name, self.vector_name)

    def get_kb_path(self):
        return get_kb_path(self.kb_name)

    def load_vector_store(self) -> ThreadSafeFaiss:
        # 从缓存中加载向量库（首次的时候缓存中没有，会先创建，然后缓存，最终我们拿到的还是缓存里面的）
        return kb_faiss_pool.load_vector_store(
            kb_name=self.kb_name,
            vector_name=self.vector_name,
            embed_model=self.embed_model,
        )

    def save_vector_store(self):
        self.load_vector_store().save(self.vs_path)

    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        with self.load_vector_store().acquire() as vs:
            return [vs.docstore._dict.get(id) for id in ids]

    def del_doc_by_ids(self, ids: List[str]) -> bool:
        with self.load_vector_store().acquire() as vs:
            vs.delete(ids)

    def do_init(self):
        self.vector_name = self.vector_name or self.embed_model.replace(":", "_")
        self.kb_path = self.get_kb_path()
        self.vs_path = self.get_vs_path()

    def do_create_kb(self):
        if not os.path.exists(self.vs_path):
            os.makedirs(self.vs_path)
        self.load_vector_store()

    def do_drop_kb(self):
        # 清空这个知识库的表（向量库）
        self.clear_vs()
        try:
            # 递归删除知识库目录及其所有内容
            shutil.rmtree(self.kb_path)
        except Exception:
            pass

    def do_search(
        self,
        query: str,
        top_k: int,
        score_threshold: float = Settings.kb_settings.SCORE_THRESHOLD,
    ) -> List[Tuple[Document, float]]:
        with self.load_vector_store().acquire() as vs:
            retriever = get_Retriever("ensemble").from_vectorstore(
                vs,
                top_k=top_k,
                score_threshold=score_threshold,
            )
            docs = retriever.get_relevant_documents(query)
        return docs

    def do_add_doc(
        self,
        docs: List[Document],
        **kwargs,
    ) -> List[Dict]:
        # 、、将docs遍历获取里面的page_content,并放入texts数组中
        texts = [x.page_content for x in docs]
        # 、、将docs遍历获取里面的metadata，放入metadatas数组中
        metadatas = [x.metadata for x in docs]
        # 、、加个类型方便跳转源文件
        vs: FAISS
        # 、、获取锁，并且加载向量库，vs 就是向量库对象，并且是线程安全的
        with self.load_vector_store().acquire()  as vs:
            """mbeddings
            # 、、调用向量库的embeddings，这个embeddings就是我们初始化加载向量库的时候传进去的embed_model，
            # 、、注意： 这个模型在传进去后经过了get_Embeddings函数的转化，
            # 、、这个get_Embeddings内部又将我们传入的embed_model传给了本地启动大模型的框架，如Ollama，然后返回一个实例化的类，
            # 、、这个实例化类中实现了一些方法，其中就包括embed_documents,add_embeddings方法，各个本地运行大模型的框架都会实现这两个方法，
            # 、、具体例子可以参考LocalAIEmbeddings这类中的实现，这个embeddings是texts向量化了的结果，具体的样子为一个二维矩阵，
            # 、、比如texts为['我爱你','你好','认真做事']，假设这个嵌入模型的嵌入层为512维
            # 、、那么embeddings就是[[0.011，0.01,.....0.1]  #长度为512，代表 我爱你
            #                      [0.012，0.02,.....0.2]  #长度为512，代表 您好
            #                      [0.013，0.03,.....0.3]  #长度为512，代表 认真做事]
            # 注意：每个文本被编码为一个完整的向量，不是按字符拆分。
            # 嵌入模型理解整个文本的语义，生成一个固定维度的向量表示。
            """
            embeddings = vs.embeddings.embed_documents(texts)
            """zip函数
             # zip函数： 将多个可迭代对象中相同位置的元素组合成元组,
            # 如：
                # texts = ["文档内容1", "文档内容2", "文档内容3"]
                # embeddings = [
                #     [0.1, 0.2, 0.3, ..., 0.512],  # 文档1的向量
                #     [0.4, 0.5, 0.6, ..., 0.512],  # 文档2的向量
                #     [0.7, 0.8, 0.9, ..., 0.512],  # 文档3的向量
                # ]
                # 使用 zip 配对
                # text_embeddings = zip(texts, embeddings)
                # zip() 后的结果
                # text_embeddings 是一个 zip 对象（迭代器）
                # 转换为列表查看：
                # list(text_embeddings) = [
                #     ("文档内容1", [0.1, 0.2, 0.3, ..., 0.512]),
                #     ("文档内容2", [0.4, 0.5, 0.6, ..., 0.512]), 
                #     ("文档内容3", [0.7, 0.8, 0.9, ..., 0.512])
                # ]
            """
            # add_embeddings方法是Faiss库中的方法,具体实现见langchain_community中的vs文件夹下faiss.py文件FAISS
            # ids 是每个文档在添加进Faiss向量库时候生成的唯一的id（uuid） 具体见FAISS的 __add()方法的第202行
            ids = vs.add_embeddings(
                text_embeddings=zip(texts, embeddings), metadatas=metadatas
            )
            if not kwargs.get("not_refresh_vs_cache"):
                # 、、上面的add_embeddings是将数据储存在了内存里面，并没有进行持久化到磁盘，save_local就是根据路径去将相应的内存里面知识库持久化到磁盘
                vs.save_local(self.vs_path)
        # 将Faiss生成的ids和docs进行zip，生成一个可迭代对象，循环这个可迭代对象将id于doc中的metadata映射到同一个对象中，并生成数组List[id,metadata]
        doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
        return doc_infos
  
    def do_delete_doc(self, kb_file: KnowledgeFile, **kwargs):  
        with self.load_vector_store().acquire() as vs:
            ids = [
                k
                for k, v in vs.docstore._dict.items()
                if v.metadata.get("source").lower() == kb_file.filename.lower()
            ]
            if len(ids) > 0:
                vs.delete(ids)
            if not kwargs.get("not_refresh_vs_cache"):
                vs.save_local(self.vs_path)
        return ids

    def do_clear_vs(self):
        with kb_faiss_pool.atomic:
            kb_faiss_pool.pop((self.kb_name, self.vector_name))
        try:
            shutil.rmtree(self.vs_path)
        except Exception:
            ...
        os.makedirs(self.vs_path, exist_ok=True)

    def exist_doc(self, file_name: str):
        if super().exist_doc(file_name):
            return "in_db"

        content_path = os.path.join(self.kb_path, "content")
        if os.path.isfile(os.path.join(content_path, file_name)):
            return "in_folder"
        else:
            return False


if __name__ == "__main__":
    faissService = FaissKBService("test")
    faissService.add_doc(KnowledgeFile("README.md", "test"))
    faissService.delete_doc(KnowledgeFile("README.md", "test"))
    faissService.do_drop_kb()
    print(faissService.search_docs("如何启动api服务"))
