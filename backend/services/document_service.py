"""
知识库管理服务：加载、切分、向量同步、上传入库、混合检索索引。

线程安全：
- 解析、切分、向量化等耗时操作在锁外执行；
- 写入 Qdrant、更新 Chunk 列表、替换检索器、检索本身都在 _lock 内，
  因为 Qdrant 本地模式和 BM25 索引替换不能被并发访问打断。
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import threading
from pathlib import Path
from typing import Any

from backend.core.config import settings
from backend.core.exceptions import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    UnprocessableDocumentError,
)
from backend.rag.embedding import EmbeddingModel
from backend.rag.loader import Document, DocumentLoader
from backend.rag.reranker import Reranker
from backend.rag.retriever import HybridRetriever
from backend.rag.splitter import TextSplitter
from backend.rag.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)

UPLOAD_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DocumentService:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: QdrantVectorStore,
        reranker: Reranker | None = None,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.reranker = reranker  # 为 None 时只做入库（scripts/ingest.py）

        self.loader = DocumentLoader()
        self.splitter = TextSplitter()

        self._chunks: list[Document] = []
        self._hashes: dict[str, str] = {}  # sha256 -> 文件名
        self._retriever: HybridRetriever | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # 启动时加载
    # ------------------------------------------------------------------
    def load_uploaded_documents(self) -> list[Document]:
        """加载 data/uploads/<upload_id>/<文件名> 下的历史上传文件。"""
        documents: list[Document] = []

        if not settings.upload_dir.exists():
            return documents

        for path in sorted(settings.upload_dir.glob("*/*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in settings.supported_extensions:
                continue

            upload_id = path.parent.name

            # 兼容旧版本存储的 "<upload_id>_原文件名"
            display_name = path.name
            if display_name.startswith(f"{upload_id}_"):
                display_name = display_name[len(upload_id) + 1:]

            try:
                documents.extend(
                    self.loader.load_file(
                        path,
                        display_name=display_name,
                        document_id=f"upload:{upload_id}",
                    )
                )
            except Exception:
                logger.exception("加载上传文件失败：%s", path)

        return documents

    def load_all_documents(self) -> list[Document]:
        """知识库目录 + 历史上传文件，不涉及模型和向量库。"""
        documents = self.loader.load_documents()
        documents.extend(self.load_uploaded_documents())
        return documents

    def _scan_hashes(self) -> dict[str, str]:
        hashes: dict[str, str] = {}

        for root in (settings.knowledge_dir, settings.upload_dir):
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in settings.supported_extensions:
                    hashes[file_sha256(path)] = path.name

        return hashes

    def _sync_vectors(self, chunks: list[Document]) -> None:
        """只给 Qdrant 里还没有的 Chunk 生成向量并写入。"""
        existing = self.vector_store.get_existing_chunk_ids()
        missing = [c for c in chunks if c.id not in existing]

        if not missing:
            logger.info("向量库已是最新（%d 条），无需写入", len(existing))
            return

        logger.info("需要补写 %d 个 Chunk 的向量", len(missing))
        vectors = self.embedding_model.encode_documents(missing)
        self.vector_store.add_documents(missing, vectors)

    def _rebuild_retriever(self) -> None:
        if self.reranker is None:
            self._retriever = None
            return

        self._retriever = HybridRetriever(
            chunks=self._chunks,
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            reranker=self.reranker,
        )

    def load_knowledge_base(self) -> None:
        documents = self.load_all_documents()
        logger.info("知识库文档总数：%d", len(documents))

        chunks = self.splitter.split_documents(documents)
        logger.info("Chunk 总数：%d", len(chunks))

        self._sync_vectors(chunks)

        hashes = self._scan_hashes()

        with self._lock:
            self._chunks = chunks
            self._hashes = hashes
            self._rebuild_retriever()

    # ------------------------------------------------------------------
    # 上传入库
    # ------------------------------------------------------------------
    def add_document(
        self,
        file_path: str | Path,
        original_name: str,
        upload_id: str,
    ) -> dict[str, Any]:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在：{file_path}")

        file_hash = file_sha256(file_path)

        # 先占位再处理，避免两个相同文件同时上传都通过检查
        with self._lock:
            if file_hash in self._hashes:
                raise DuplicateDocumentError(
                    f"该文件内容已经在知识库中（{self._hashes[file_hash]}）"
                )
            self._hashes[file_hash] = original_name

        try:
            try:
                documents = self.loader.load_file(
                    file_path,
                    display_name=original_name,
                    document_id=f"upload:{upload_id}",
                )
            except Exception as e:
                raise UnprocessableDocumentError(f"文件解析失败：{e}") from e

            if not documents:
                raise UnprocessableDocumentError(
                    "没有从文件中解析出有效内容（扫描版 PDF 需要先做 OCR）"
                )

            new_chunks = self.splitter.split_documents(documents)

            if not new_chunks:
                raise UnprocessableDocumentError("文档切分后没有生成有效内容")

            vectors = self.embedding_model.encode_documents(new_chunks)

            with self._lock:
                self.vector_store.add_documents(new_chunks, vectors)
                self._chunks.extend(new_chunks)
                self._rebuild_retriever()
                vector_count = self.vector_store.count()

        except Exception:
            with self._lock:
                self._hashes.pop(file_hash, None)
            raise

        return {
            "file_name": original_name,
            "document_count": len(documents),
            "chunk_count": len(new_chunks),
            "vector_count": vector_count,
            "message": "文档上传并加入知识库成功",
        }

    # ------------------------------------------------------------------
    # 列表与删除
    # ------------------------------------------------------------------
    def list_documents(self) -> list[dict[str, Any]]:
        """按文档汇总当前知识库（PDF 的多页合并成一份）。"""
        with self._lock:
            groups: dict[str, dict[str, Any]] = {}

            for chunk in self._chunks:
                meta = chunk.metadata
                document_id = meta.get("document_id", chunk.id)
                uploaded = document_id.startswith("upload:")

                info = groups.setdefault(
                    document_id,
                    {
                        "document_id": document_id,
                        "file_name": meta.get("file_name", ""),
                        "title": meta.get("title", ""),
                        "chunk_count": 0,
                        "deletable": uploaded,
                        "upload_id": document_id.split(":", 1)[1] if uploaded else None,
                    },
                )
                info["chunk_count"] += 1

            return sorted(
                groups.values(), key=lambda d: (d["deletable"], d["document_id"])
            )

    def delete_document(self, upload_id: str) -> dict[str, Any]:
        """
        删除一份上传的文档：向量、内存中的 Chunk、BM25 索引、磁盘文件和去重记录一起清理。
        只允许删除通过上传接口加入的文档。
        """
        if not UPLOAD_ID_PATTERN.fullmatch(upload_id):
            raise DocumentNotFoundError(f"文档不存在：{upload_id}")

        document_id = f"upload:{upload_id}"
        upload_dir = settings.upload_dir / upload_id

        with self._lock:
            doomed = [
                c for c in self._chunks if c.metadata.get("document_id") == document_id
            ]

            if not doomed and not upload_dir.exists():
                raise DocumentNotFoundError(f"文档不存在：{upload_id}")

            files = (
                [p for p in upload_dir.iterdir() if p.is_file()]
                if upload_dir.exists()
                else []
            )
            file_name = (
                doomed[0].metadata.get("file_name")
                if doomed
                else (files[0].name if files else upload_id)
            )

            # 先删向量；失败会直接抛出，此时其余状态都没有改动
            self.vector_store.delete_by_ids([c.id for c in doomed])

            doomed_ids = {c.id for c in doomed}
            self._chunks = [c for c in self._chunks if c.id not in doomed_ids]

            # 释放去重记录，之后允许重新上传同一份文件
            for path in files:
                self._hashes.pop(file_sha256(path), None)

            shutil.rmtree(upload_dir, ignore_errors=True)
            self._rebuild_retriever()

            return {
                "upload_id": upload_id,
                "file_name": file_name,
                "deleted_chunks": len(doomed),
                "vector_count": self.vector_store.count(),
                "message": "文档已从知识库删除",
            }

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int, candidate_k: int) -> list[dict[str, Any]]:
        with self._lock:
            if self._retriever is None:
                raise RuntimeError("检索器尚未初始化")
            return self._retriever.retrieve(
                query=query, top_k=top_k, candidate_k=candidate_k
            )

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "chunk_count": len(self._chunks),
                "vector_count": self.vector_store.count(),
            }
