
"""
RAG Retriever
负责将用户问题转换为向量，并从 Qdrant 检索相关知识。
"""

from typing import List, Dict, Any

from embedding import EmbeddingModel
from vector_store import QdrantVectorStore


class Retriever:

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: QdrantVectorStore,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        根据用户问题检索相关 Chunk。
        """

        if not query or not query.strip():
            raise ValueError("用户问题不能为空")

        # 1. 将问题转换为向量
        query_vector = self.embedding_model.encode_query(
            query
        )

        # 2. 调用 Qdrant 进行相似度搜索
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
        )

        return results


if __name__ == "__main__":

    print("=" * 50)
    print("RAG Retriever 模块测试")
    print("=" * 50)

    embedding_model = EmbeddingModel(
        model_name="BAAI/bge-m3",
        device="auto",
        batch_size=8,
    )

    vector_store = QdrantVectorStore(
        dimension=embedding_model.dimension
    )

    try:
        retriever = Retriever(
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        query = "星云科技的研发环境需要安装什么 Python 版本？"

        print(f"\n用户问题：{query}")

        results = retriever.retrieve(
            query=query,
            top_k=3,
        )

        print("\n检索结果：")

        for index, result in enumerate(
            results,
            start=1,
        ):
            metadata = result["metadata"]

            print("\n" + "-" * 50)
            print(f"排名：{index}")
            print(f"相似度：{result['score']:.4f}")
            print(f"Chunk ID：{result['chunk_id']}")
            print(f"文档标题：{metadata.get('title', '未知')}")
            print(f"来源：{metadata.get('source', '未知')}")
            print(f"章节：{metadata.get('heading', '未知')}")
            print(f"内容：\n{result['content'][:500]}")

        print("\nRetriever 测试完成！")

    finally:
        vector_store.close()
