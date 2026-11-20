"""BGE-M3 向量化：批量编码文本，输出已归一化的 float32 向量。"""

from __future__ import annotations

import logging

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from backend.core.config import settings
from backend.rag.loader import Document

logger = logging.getLogger(__name__)


class EmbeddingModel:
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ):
        self.model_name = model_name or settings.embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size

        device = device or settings.embedding_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info("正在加载 Embedding 模型：%s（设备：%s）", self.model_name, self.device)
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info("Embedding 模型加载成功，向量维度：%d", self.dimension)

    def encode_texts(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("texts 中不能包含空文本或非字符串")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size or self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        if not query or not query.strip():
            raise ValueError("查询内容不能为空")
        return self.encode_texts([query])[0]

    def encode_documents(self, documents: list[Document]) -> np.ndarray:
        # 批量入库才显示进度条，单次提问不刷屏
        return self.encode_texts(
            [d.content for d in documents], show_progress=True
        )
