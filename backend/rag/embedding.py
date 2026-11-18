
"""
RAG Embedding
1. 加载本地 BGE-M3 模型
2. 将文本转换为 1024 维向量
3. 支持批量文本向量化
4. 支持 Document 对象向量化
5. 对向量进行归一化，方便后续相似度检索
"""

from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from loader import Document
from splitter import TextSplitter
from loader import load_documents


class EmbeddingModel:

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        batch_size: int = 8,
    ):
        self.model_name = model_name
        self.batch_size = batch_size

        # 自动选择运行设备
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        print(f"正在加载 Embedding 模型：{model_name}")
        print(f"运行设备：{self.device}")

        self.model = SentenceTransformer(
            model_name,
            device=self.device,
        )

        # BGE-M3 的向量维度为 1024
        self.dimension = self.model.get_sentence_embedding_dimension()

        print("模型加载成功！")
        print(f"向量维度：{self.dimension}")

    def encode_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        将多个文本转换为向量。

        返回：
            numpy.ndarray，形状为
            (文本数量, 向量维度)
        """

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts 中不能包含空文本或非字符串")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size or self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        将用户问题转换为向量。
        """

        if not query or not query.strip():
            raise ValueError("查询内容不能为空")

        return self.encode_texts([query])[0]

    def encode_documents(
        self,
        documents: List[Document],
    ) -> np.ndarray:
        """
        将 Document 列表转换为向量。
        """

        texts = [document.content for document in documents]

        return self.encode_texts(texts)


if __name__ == "__main__":

    print("=" * 50)
    print("RAG Embedding 模块测试")
    print("=" * 50)

    # 1. 加载原始文档
    documents = load_documents()
    print(f"\n原始文档数量：{len(documents)}")

    # 2. 文本切分
    splitter = TextSplitter(
        chunk_size=600,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(documents)
    print(f"Chunk 数量：{len(chunks)}")

    # 3. 加载 Embedding 模型
    embedding_model = EmbeddingModel(
        model_name="BAAI/bge-m3",
        device="auto",
        batch_size=8,
    )

    # 4. 将 Chunk 转换为向量
    embeddings = embedding_model.encode_documents(chunks)

    # 5. 检查结果
    print("\n" + "=" * 50)
    print("Embedding 测试结果")
    print("=" * 50)

    print(f"向量数量：{len(embeddings)}")
    print(f"向量维度：{embeddings.shape[1]}")
    print(f"第一个向量前 10 个值：{embeddings[0][:10]}")

    # 6. 检查向量归一化
    vector_norm = np.linalg.norm(embeddings[0])
    print(f"第一个向量的模长：{vector_norm:.4f}")

    print("\nEmbedding 模块测试完成！")
