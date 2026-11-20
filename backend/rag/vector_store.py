"""Qdrant 本地持久化向量库：写入、检索，以及增量同步需要的查询。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

from backend.core.config import settings
from backend.rag.loader import Document

logger = logging.getLogger(__name__)


def make_qdrant_id(chunk_id: str) -> str:
    """把 Chunk ID 转成确定性 UUID（Qdrant 要求合法 Point ID）。"""
    return str(uuid5(NAMESPACE_URL, chunk_id))


def make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    return str(value)


class QdrantVectorStore:
    def __init__(
        self,
        path: str | Path | None = None,
        collection_name: str | None = None,
        dimension: int = 1024,
    ):
        self.path = str(path or settings.qdrant_path)
        self.collection_name = collection_name or settings.collection_name
        self.dimension = dimension

        Path(self.path).mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=self.path)
        logger.info("Qdrant 本地数据库连接成功：%s", self.path)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension, distance=Distance.COSINE
                ),
            )
            logger.info("Collection 创建成功：%s", self.collection_name)
            return

        info = self.client.get_collection(self.collection_name)
        size = info.config.params.vectors.size

        if size != self.dimension:
            raise ValueError(
                f"Collection 向量维度不匹配：数据库为 {size}，当前模型为 {self.dimension}"
            )

        logger.info("Collection 已存在：%s", self.collection_name)

    def reset(self) -> None:
        """删除并重建 Collection（scripts/ingest.py --rebuild 使用）。"""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self._ensure_collection()

    def get_existing_chunk_ids(self) -> set[str]:
        """返回库中已有的全部 chunk_id，用于增量同步。"""
        ids: set[str] = set()
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=["chunk_id"],
                with_vectors=False,
            )
            ids.update(
                p.payload["chunk_id"]
                for p in points
                if p.payload and p.payload.get("chunk_id")
            )
            if offset is None:
                break

        return ids

    def delete_by_ids(self, chunk_ids: list[str]) -> None:
        """按 chunk_id 删除向量（文档删除时使用）。"""
        if not chunk_ids:
            return

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(
                points=[make_qdrant_id(chunk_id) for chunk_id in chunk_ids]
            ),
            wait=True,
        )
        logger.info("已删除 %d 条向量", len(chunk_ids))

    def add_documents(
        self,
        documents: list[Document],
        embeddings: np.ndarray,
        batch_size: int = 32,
    ) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("文档数量与向量数量不一致")

        if len(documents) == 0:
            return

        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"向量形状错误：预期 (N, {self.dimension})，实际 {embeddings.shape}"
            )

        points = [
            PointStruct(
                id=make_qdrant_id(doc.id),
                vector=vector.tolist(),
                payload={
                    "chunk_id": doc.id,
                    "content": doc.content,
                    "metadata": make_json_safe(doc.metadata),
                },
            )
            for doc, vector in zip(documents, embeddings)
        ]

        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[start:start + batch_size],
                wait=True,
            )

        logger.info("向量写入完成：%d 条", len(points))

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        if query_vector.shape != (self.dimension,):
            raise ValueError(f"查询向量维度错误，预期为 {self.dimension}")

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        )

        results = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "content": payload.get("content"),
                    "metadata": payload.get("metadata", {}),
                    "score": point.score,
                }
            )
        return results

    def count(self) -> int:
        return self.client.count(
            collection_name=self.collection_name, exact=True
        ).count

    def close(self) -> None:
        self.client.close()
