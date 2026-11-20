"""CrossEncoder 重排序：对候选结果按与问题的相关性重新打分。"""

from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

from backend.core.config import settings

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        max_length: int | None = None,
    ):
        self.model_name = model_name or settings.reranker_model

        logger.info("正在加载 Reranker 模型：%s", self.model_name)
        self.model = CrossEncoder(
            self.model_name,
            device=device or settings.reranker_device,
            max_length=max_length or settings.reranker_max_length,
        )
        logger.info("Reranker 模型加载成功")

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 3,
    ) -> list[dict]:
        """
        candidates: 每项是带 "content" 的字典。
        返回按 rerank_score（Sigmoid 后 0~1）从高到低的前 top_k 项。
        """
        if not query.strip():
            raise ValueError("查询问题不能为空")

        if not candidates:
            return []

        pairs = [[query, item.get("content") or ""] for item in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        results = [
            {**item, "rerank_score": float(score)}
            for item, score in zip(candidates, scores)
        ]
        results.sort(key=lambda x: x["rerank_score"], reverse=True)

        return results[:top_k]
