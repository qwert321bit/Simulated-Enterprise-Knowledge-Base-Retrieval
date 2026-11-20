"""问答服务：检索 -> 相关性判断 -> 生成答案（一次返回或流式）。"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from backend.core.config import settings
from backend.rag.generator import NO_ANSWER, RAGGenerator
from backend.services.document_service import DocumentService

logger = logging.getLogger(__name__)

LOW_RELEVANCE_ANSWER = "检索到的知识库资料与问题相关性较低，暂时无法给出可靠答案。"


@dataclass
class StreamResult:
    """流式问答的结果：来源已确定，答案通过 tokens 逐段产出。"""

    query: str
    sources: list[dict[str, Any]]
    retrieved_count: int
    tokens: Iterator[str]


class ChatService:
    def __init__(self, document_service: DocumentService, generator: RAGGenerator):
        self.documents = document_service
        self.generator = generator

    def _prepare(
        self,
        query: str,
        top_k: int | None,
        candidate_k: int | None,
    ) -> tuple[str, list[dict[str, Any]], str | None]:
        """检索并判断是否拒答。返回 (规整后的问题, 检索结果, 拒答文案或 None)。"""
        query = (query or "").strip()

        if not query:
            raise ValueError("问题不能为空")

        top_k = top_k or settings.default_top_k
        candidate_k = max(candidate_k or settings.candidate_k, top_k)

        logger.info("用户问题：%s", query)

        retrieved = self.documents.retrieve(query, top_k, candidate_k)

        if not retrieved:
            return query, retrieved, NO_ANSWER

        # 阈值是初始实验值，需要用评测集校准
        best_score = retrieved[0].get("rerank_score")

        if best_score is not None and best_score < settings.min_rerank_score:
            logger.info("最高 rerank 分数 %.3f 低于阈值，拒答", best_score)
            return query, retrieved, LOW_RELEVANCE_ANSWER

        return query, retrieved, None

    def ask(
        self,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
    ) -> dict[str, Any]:
        query, retrieved, refusal = self._prepare(query, top_k, candidate_k)

        if refusal is not None:
            return {
                "query": query,
                "answer": refusal,
                "sources": [],
                "retrieved_chunks": retrieved,
            }

        result = self.generator.generate(query=query, retrieved_chunks=retrieved)

        return {
            "query": query,
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieved_chunks": retrieved,
        }

    def ask_stream(
        self,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
    ) -> StreamResult:
        """
        流式问答。检索和拒答判断在这里同步完成（出错会直接抛出，
        API 层可以返回正常的 HTTP 错误码），答案文本由 tokens 逐段产出。
        """
        query, retrieved, refusal = self._prepare(query, top_k, candidate_k)

        if refusal is not None:
            return StreamResult(query, [], len(retrieved), iter([refusal]))

        sources, tokens = self.generator.stream(query=query, retrieved_chunks=retrieved)

        return StreamResult(query, sources, len(retrieved), tokens)
