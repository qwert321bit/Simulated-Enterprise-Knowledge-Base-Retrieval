"""
混合检索：BGE-M3 向量检索 + BM25 关键词检索 -> RRF 融合 -> Reranker 重排。

HybridRetriever 构造后索引不可变；有新文档时由 DocumentService 重新构造并替换。
"""

from __future__ import annotations

import logging
import re
from typing import Any

import jieba
from rank_bm25 import BM25Okapi

from backend.core.config import settings
from backend.rag.embedding import EmbeddingModel
from backend.rag.loader import Document
from backend.rag.reranker import Reranker
from backend.rag.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """中文用 jieba，英文/数字/技术名词用正则保留（如 python3.11、c++）。"""
    tokens = []

    for word in jieba.lcut(text.lower()):
        word = word.strip()
        if not word:
            continue
        is_ascii_term = re.fullmatch(r"[a-z0-9_.+#/-]+", word) and re.search(r"[a-z0-9]", word)

        # 纯标点（".", "-", "/"）不是检索词，会给每个含标点的 Chunk 加噪声分
        if is_ascii_term or re.search(r"[\u4e00-\u9fff]", word):
            tokens.append(word)

    return tokens


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Document],
        embedding_model: EmbeddingModel,
        vector_store: QdrantVectorStore,
        reranker: Reranker,
        rrf_k: int | None = None,
    ):
        self.chunks = list(chunks)
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.reranker = reranker
        self.rrf_k = rrf_k or settings.rrf_k

        self.chunk_map = {chunk.id: chunk for chunk in self.chunks}

        # 知识库为空时 BM25Okapi 会除零，此时只走向量检索
        corpus = [tokenize(chunk.content) for chunk in self.chunks]
        self.bm25 = BM25Okapi(corpus) if any(corpus) else None

        logger.info("BM25 索引建立成功，共 %d 个 Chunk", len(self.chunks))

    def _bm25_search(self, query: str, k: int) -> list[str]:
        """返回 BM25 得分 > 0 的前 k 个 chunk_id。"""
        if self.bm25 is None:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        return [self.chunks[i].id for i in ranked if scores[i] > 0]

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not query or not query.strip():
            raise ValueError("查询内容不能为空")

        top_k = top_k or settings.default_top_k
        candidate_k = max(candidate_k or settings.candidate_k, top_k)

        # 1. 向量检索
        query_vector = self.embedding_model.encode_query(query)
        vector_results = self.vector_store.search(query_vector, top_k=candidate_k)
        vector_by_id = {r["chunk_id"]: r for r in vector_results if r["chunk_id"]}

        # 2. BM25 检索
        bm25_ids = self._bm25_search(query, candidate_k)

        # 3. RRF 融合
        fused: dict[str, float] = {}
        details: dict[str, dict] = {}

        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result["chunk_id"]
            if not chunk_id:
                continue
            fused[chunk_id] = fused.get(chunk_id, 0) + 1 / (self.rrf_k + rank)
            details.setdefault(chunk_id, {}).update(
                vector_score=result["score"], vector_rank=rank
            )

        for rank, chunk_id in enumerate(bm25_ids, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0) + 1 / (self.rrf_k + rank)
            details.setdefault(chunk_id, {})["bm25_rank"] = rank

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:candidate_k]

        # 4. 组装 Reranker 候选
        candidates = []

        for chunk_id, rrf_score in ranked:
            chunk = self.chunk_map.get(chunk_id)

            if chunk is not None:
                content, metadata = chunk.content, chunk.metadata
            elif chunk_id in vector_by_id:
                # Qdrant 里有、本地索引里没有的记录
                content = vector_by_id[chunk_id]["content"]
                metadata = vector_by_id[chunk_id]["metadata"]
            else:
                continue

            candidates.append(
                {
                    "chunk_id": chunk_id,
                    "content": content,
                    "metadata": metadata,
                    "rrf_score": rrf_score,
                    "details": details[chunk_id],
                }
            )

        # 5. 重排序
        return self.reranker.rerank(query=query, candidates=candidates, top_k=top_k)
