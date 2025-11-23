from __future__ import annotations

from langchain.retrievers import EnsembleRetriever
from langchain.vectorstores import VectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever

from chatchat.server.file_rag.retrievers.base import BaseRetrieverService


class EnsembleRetrieverService(BaseRetrieverService):
    def do_init(
        self,
        retriever: BaseRetriever = None,
        top_k: int = 5,
    ):
        self.vs = None # 向量库
        self.top_k = top_k # 返回结果数量
        self.retriever = retriever # 
    
    # // 这是一个静态方法 所以可以被直接不用实例化类就调用,在FiassKBService中的do_search方法,直接调用这个方法生成一个实例,
    # 从实例上调用 get_relevant_documents 获取搜索结果
    @staticmethod
    def from_vectorstore(
        vectorstore: VectorStore,
        top_k: int,
        score_threshold: int | float,
    ):
        # 、、Faiss向量检索器，基于语义相似度
        faiss_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold", #使用分数阈值
            search_kwargs={
                "score_threshold": score_threshold, #相似度阈值，过滤低质量结果
                "k": top_k # 返回结果数量
            },
        )
        # TODO: 换个不用torch的实现方式
        # from cutword.cutword import Cutter
        import jieba
        # jieba是一个比较流行的中文分词库,用于将中文文本切分成有意义的词语

        # cutter = Cutter()
        # 、、知识库中获取所有分割过的docs，利用bm25进行关键词匹配，使用jieba对doc中的page_content进行分词
        docs = list(vectorstore.docstore._dict.values())
        bm25_retriever = BM25Retriever.from_documents(
            docs,
            preprocess_func=jieba.lcut_for_search,
        )
        # 、、设置返回结果数量
        bm25_retriever.k = top_k
        # 创建组合检索器，融合两种检索策略，
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], 
            weights=[0.5, 0.5]# 两种检索器结果各占50权重
        )
        # 创建并返回检索器服务实例
        return EnsembleRetrieverService(retriever=ensemble_retriever, top_k=top_k)

    def get_relevant_documents(self, query: str):
        return self.retriever.get_relevant_documents(query)[: self.top_k]
