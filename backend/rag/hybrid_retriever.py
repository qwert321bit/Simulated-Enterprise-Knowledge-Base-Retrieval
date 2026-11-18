
"""
Hybrid Retriever
1. BGE-M3 + Qdrant 向量检索
2. BM25 关键词检索
3. RRF 融合排序
"""

import re
from typing import List, Dict, Any

import jieba
from rank_bm25 import BM25Okapi

from loader import load_documents
from splitter import TextSplitter
from embedding import EmbeddingModel
from vector_store import QdrantVectorStore


def tokenize(text: str) -> List[str]:
    """
    中英文混合分词：
    - 中文使用 jieba
    - 英文、数字和技术名词使用正则提取
    """

    text = text.lower()

    words = jieba.lcut(text)

    tokens = []

    for word in words:
        word = word.strip()

        if not word:
            continue

        # 保留英文、数字、版本号和技术名词
        if re.fullmatch(r"[a-z0-9_.+#/-]+", word):
            tokens.append(word)

        # 保留中文词语
        elif re.search(r"[\u4e00-\u9fff]", word):
            tokens.append(word)

    return tokens


class HybridRetriever:

    def __init__(
        self,
        chunks,
        embedding_model,
        vector_store,
        rrf_k: int = 60,
    ):
        self.chunks = chunks
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.rrf_k = rrf_k

        # 建立 BM25 索引
        corpus = [
            tokenize(chunk.content)
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(corpus)

        # 根据 Chunk ID 建立映射
        self.chunk_map = {
            chunk.id: chunk
            for chunk in chunks
        }

        print(f"BM25 索引建立成功，共 {len(chunks)} 个 Chunk")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10,
    ) -> List[Dict[str, Any]]:

        if not query or not query.strip():
            raise ValueError("查询内容不能为空")

        # 1. 向量检索
        query_vector = self.embedding_model.encode_query(
            query
        )

        vector_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=candidate_k,
        )

        # 2. BM25 检索
        query_tokens = tokenize(query)

        bm25_scores = self.bm25.get_scores(query_tokens)

        bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:candidate_k]

        bm25_results = [
            self.chunks[i]
            for i in bm25_indices
            if bm25_scores[i] > 0
        ]

        # 3. RRF 融合
        fused_scores = {}

        details = {}

        # 向量检索排名
        for rank, result in enumerate(
            vector_results,
            start=1,
        ):
            chunk_id = result["chunk_id"]

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0)
                + 1 / (self.rrf_k + rank)
            )

            details.setdefault(chunk_id, {})
            details[chunk_id]["vector_score"] = result["score"]
            details[chunk_id]["vector_rank"] = rank

        # BM25 排名
        for rank, chunk in enumerate(
            bm25_results,
            start=1,
        ):
            chunk_id = chunk.id

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0)
                + 1 / (self.rrf_k + rank)
            )

            details.setdefault(chunk_id, {})
            details[chunk_id]["bm25_rank"] = rank

        # 4. 按融合分数排序
        sorted_results = sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        results = []

        for chunk_id, score in sorted_results:

            # 优先从本地 Chunk 映射获取完整信息
            chunk = self.chunk_map.get(chunk_id)

            if chunk is not None:
                content = chunk.content
                metadata = chunk.metadata

            else:
                # 兼容 Qdrant 中存在但本地索引中没有的记录
                result = next(
                    item for item in vector_results
                    if item["chunk_id"] == chunk_id
                )

                content = result["content"]
                metadata = result["metadata"]

            results.append({
                "chunk_id": chunk_id,
                "content": content,
                "metadata": metadata,
                "rrf_score": score,
                "details": details[chunk_id],
            })

        return results


if __name__ == "__main__":

    print("=" * 50)
    print("Hybrid Retriever 测试")
    print("=" * 50)

    # 加载文档
    documents = load_documents()

    # 切分
    splitter = TextSplitter(
        chunk_size=600,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(documents)

    print(f"Chunk 数量：{len(chunks)}")

    # 加载 Embedding
    embedding_model = EmbeddingModel(
        model_name="BAAI/bge-m3",
        device="auto",
        batch_size=8,
    )

    # 连接现有 Qdrant 数据库
    vector_store = QdrantVectorStore(
        dimension=embedding_model.dimension
    )

    try:
        # 建立混合检索器
        retriever = HybridRetriever(
            chunks=chunks,
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        # 测试问题
        queries = [
            "星云科技的研发环境需要安装什么 Python 版本？",
            "PostgreSQL 16 数据库使用什么规范？",
            "Qdrant 向量数据库如何部署？",
        ]

        for query in queries:

            print("\n" + "=" * 50)
            print(f"问题：{query}")

            results = retriever.retrieve(
                query=query,
                top_k=3,
                candidate_k=10,
            )

            for index, result in enumerate(
                results,
                start=1,
            ):
                print("\n" + "-" * 40)
                print(f"排名：{index}")
                print(f"Chunk：{result['chunk_id']}")
                print(f"RRF 分数：{result['rrf_score']:.6f}")
                print(f"检索详情：{result['details']}")
                print(f"标题：{result['metadata'].get('title')}")
                print(f"内容：\n{result['content'][:300]}")

        print("\nHybrid Retriever 测试完成！")

    finally:
        vector_store.close()
