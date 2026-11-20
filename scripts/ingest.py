"""
离线建库：把 data/enterprise 和 data/uploads 里的文档同步到 Qdrant。

    python -m scripts.ingest             # 只补写缺失的 Chunk
    python -m scripts.ingest --rebuild   # 清空向量库后全部重建

注意：Qdrant 本地模式是文件锁，运行脚本前先停掉 uvicorn。
"""

import argparse
import logging

from backend.core.config import settings
from backend.rag.embedding import EmbeddingModel
from backend.rag.vector_store import QdrantVectorStore
from backend.services.document_service import DocumentService


def main() -> None:
    parser = argparse.ArgumentParser(description="知识库离线入库")
    parser.add_argument("--rebuild", action="store_true", help="清空向量库后重建")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings.ensure_dirs()

    embedding_model = EmbeddingModel()
    vector_store = QdrantVectorStore(dimension=embedding_model.dimension)

    try:
        if args.rebuild:
            vector_store.reset()
            print("已清空向量库")

        service = DocumentService(embedding_model, vector_store, reranker=None)
        service.load_knowledge_base()

        stats = service.stats()
        print(f"入库完成：Chunk {stats['chunk_count']} 个，向量 {stats['vector_count']} 条")
    finally:
        vector_store.close()


if __name__ == "__main__":
    main()
