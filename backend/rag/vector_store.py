
"""
RAG Vector Store
1. 使用 Qdrant 本地持久化存储向量
2. 批量写入 Chunk 和 Embedding
3. 支持语义相似度检索
4. 保存原始文本和文档元数据
5. 支持查询和统计
"""

from pathlib import Path
from typing import List, Dict, Any
from uuid import uuid5, NAMESPACE_URL

import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from loader import Document
from splitter import TextSplitter
from loader import load_documents
from embedding import EmbeddingModel


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Qdrant 本地数据库保存位置
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant_storage"

# Collection 名称
COLLECTION_NAME = "enterprise_knowledge"


def make_qdrant_id(chunk_id: str) -> str:
    """
    将原始 Chunk ID 转换为确定性 UUID。

    例如：
    E001:chunk:0
    转换为合法的 UUID。
    """
    return str(uuid5(NAMESPACE_URL, chunk_id))


def make_json_safe(value):
    """
    将 metadata 转换为 Qdrant 可保存的 JSON 类型。
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(k): make_json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]

    return str(value)


class QdrantVectorStore:

    def __init__(
        self,
        path: str = str(QDRANT_PATH),
        collection_name: str = COLLECTION_NAME,
        dimension: int = 1024,
    ):
        self.path = path
        self.collection_name = collection_name
        self.dimension = dimension

        # 创建数据库目录
        Path(self.path).mkdir(
            parents=True,
            exist_ok=True,
        )

        # 连接本地 Qdrant
        self.client = QdrantClient(
            path=self.path
        )

        print("Qdrant 本地数据库连接成功！")

        self._ensure_collection()

    def _ensure_collection(self):
        """
        如果 Collection 不存在，则创建。
        如果已经存在，则检查向量维度。
        """

        if not self.client.collection_exists(
            self.collection_name
        ):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE,
                ),
            )

            print(
                f"Collection 创建成功：{self.collection_name}"
            )

        else:
            info = self.client.get_collection(
                self.collection_name
            )

            vectors_config = info.config.params.vectors

            # 当前项目使用单一向量配置
            if vectors_config.size != self.dimension:
                raise ValueError(
                    f"Collection 向量维度不匹配："
                    f"数据库为 {vectors_config.size}，"
                    f"当前模型为 {self.dimension}"
                )

            print(
                f"Collection 已存在：{self.collection_name}"
            )

    def add_documents(
        self,
        documents: List[Document],
        embeddings: np.ndarray,
        batch_size: int = 32,
    ):
        """
        批量将 Chunk 和向量写入 Qdrant。
        """

        if len(documents) != len(embeddings):
            raise ValueError(
                "文档数量与向量数量不一致"
            )

        if len(documents) == 0:
            print("没有需要写入的文档。")
            return

        if embeddings.ndim != 2:
            raise ValueError(
                "embeddings 必须是二维数组"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"向量维度错误："
                f"预期 {self.dimension}，"
                f"实际 {embeddings.shape[1]}"
            )

        points = []

        for document, vector in zip(
            documents,
            embeddings,
        ):

            # Qdrant 使用合法 UUID 作为 Point ID
            point_id = make_qdrant_id(document.id)

            # 保存原始文本及元数据
            payload = {
                "chunk_id": document.id,
                "content": document.content,
                "metadata": make_json_safe(
                    document.metadata
                ),
            }

            point = PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=payload,
            )

            points.append(point)

        # 分批写入数据库
        for start in range(
            0,
            len(points),
            batch_size,
        ):
            batch = points[
                start:start + batch_size
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True,
            )

            print(
                f"已写入 {min(start + batch_size, len(points))}"
                f"/{len(points)} 条数据"
            )

        print("向量数据写入完成！")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        根据问题向量检索最相似的 Chunk。
        """

        if query_vector.shape != (self.dimension,):
            raise ValueError(
                f"查询向量维度错误，"
                f"预期为 {self.dimension}"
            )

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

            results.append({
                "chunk_id": payload.get("chunk_id"),
                "content": payload.get("content"),
                "metadata": payload.get("metadata", {}),
                "score": point.score,
            })

        return results

    def count(self) -> int:
        """
        获取数据库中的向量数量。
        """

        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )

        return result.count

    def close(self):
        """
        关闭数据库连接。
        """

        self.client.close()


if __name__ == "__main__":

    print("=" * 50)
    print("RAG Vector Store 模块测试")
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

    # 4. 生成向量
    embeddings = embedding_model.encode_documents(
        chunks
    )

    print(f"向量数量：{len(embeddings)}")
    print(f"向量维度：{embeddings.shape[1]}")

    # 5. 创建 Qdrant 数据库
    vector_store = QdrantVectorStore(
        dimension=embedding_model.dimension
    )

    try:
        # 6. 写入向量
        vector_store.add_documents(
            documents=chunks,
            embeddings=embeddings,
        )

        print(
            f"\n数据库当前向量数量："
            f"{vector_store.count()}"
        )

        # 7. 测试语义检索
        query = "星云科技的研发环境需要安装什么 Python 版本？"

        print(f"\n测试问题：{query}")

        query_vector = embedding_model.encode_query(
            query
        )

        results = vector_store.search(
            query_vector=query_vector,
            top_k=3,
        )

        print("\n检索结果：")

        for index, result in enumerate(
            results,
            start=1,
        ):
            print("\n" + "-" * 50)

            print(f"排名：{index}")
            print(f"Chunk ID：{result['chunk_id']}")
            print(f"相似度：{result['score']:.4f}")

            metadata = result["metadata"]

            print(
                f"文档标题："
                f"{metadata.get('title', '未知')}"
            )

            print(
                f"来源："
                f"{metadata.get('source', '未知')}"
            )

            print(
                f"章节："
                f"{metadata.get('heading', '未知')}"
            )

            print(
                f"内容：\n{result['content'][:500]}"
            )

        print("\nVector Store 测试完成！")

    finally:
        vector_store.close()
