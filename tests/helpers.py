"""测试用的假对象：不加载任何模型，也不需要网络和 API Key。"""

from __future__ import annotations

import uuid
import zlib
from pathlib import Path

import numpy as np

from backend.core.config import settings
from backend.rag.loader import Document

DIM = 16


def text_to_vector(text: str) -> np.ndarray:
    """同一段文本永远得到同一个单位向量。"""
    rng = np.random.default_rng(zlib.crc32(text.encode("utf-8")))
    vec = rng.standard_normal(DIM).astype(np.float32)
    return vec / np.linalg.norm(vec)


class FakeEmbedding:
    dimension = DIM

    def __init__(self):
        self.document_calls = 0

    def encode_query(self, query: str) -> np.ndarray:
        if not query or not query.strip():
            raise ValueError("查询内容不能为空")
        return text_to_vector(query)

    def encode_documents(self, documents: list[Document]) -> np.ndarray:
        self.document_calls += 1
        if not documents:
            return np.empty((0, DIM), dtype=np.float32)
        return np.stack([text_to_vector(d.content) for d in documents])


class FakeVectorStore:
    dimension = DIM

    def __init__(self):
        self.items: dict[str, tuple[np.ndarray, Document]] = {}

    def add_documents(self, documents, embeddings, batch_size: int = 32):
        for doc, vec in zip(documents, embeddings):
            self.items[doc.id] = (np.asarray(vec), doc)

    def search(self, query_vector, top_k: int = 5):
        scored = sorted(
            ((float(np.dot(query_vector, vec)), doc) for vec, doc in self.items.values()),
            key=lambda pair: pair[0],
            reverse=True,
        )[:top_k]
        return [
            {"chunk_id": d.id, "content": d.content, "metadata": d.metadata, "score": s}
            for s, d in scored
        ]

    def count(self) -> int:
        return len(self.items)

    def get_existing_chunk_ids(self) -> set[str]:
        return set(self.items)

    def delete_by_ids(self, chunk_ids: list[str]) -> None:
        for chunk_id in chunk_ids:
            self.items.pop(chunk_id, None)

    def reset(self) -> None:
        self.items.clear()

    def close(self) -> None:
        pass


class FakeReranker:
    """用字符重合度模拟相关性，保证结果可预测。"""

    def rerank(self, query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
        if not candidates:
            return []

        query_chars = set(query)
        results = [
            {
                **c,
                "rerank_score": len(query_chars & set(c["content"])) / max(len(query_chars), 1),
            }
            for c in candidates
        ]
        results.sort(key=lambda r: r["rerank_score"], reverse=True)
        return results[:top_k]


class FakeLLM:
    model = "fake-model"

    def __init__(self, answer: str = "这是答案 [1]", tokens: list[str] | None = None):
        self.answer = answer
        self.tokens = tokens if tokens is not None else ["这是", "答案 ", "[1]"]
        self.calls: list[dict] = []

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        self.calls.append({"system": system_prompt, "user": user_prompt})
        return self.answer

    def stream_chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2):
        self.calls.append({"system": system_prompt, "user": user_prompt})
        yield from self.tokens


def make_chunk(chunk_id: str, content: str, **metadata) -> Document:
    metadata.setdefault("title", chunk_id)
    metadata.setdefault("document_id", chunk_id.split(":chunk")[0])
    return Document(id=chunk_id, content=content, metadata=metadata)


def save_upload(name: str, content: bytes) -> tuple[str, Path]:
    """模拟上传接口落盘：data/uploads/<upload_id>/<原文件名>。"""
    upload_id = uuid.uuid4().hex
    directory = settings.upload_dir / upload_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(content)
    return upload_id, path
