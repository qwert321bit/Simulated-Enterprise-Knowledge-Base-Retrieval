"""
导出当前知识库的 Chunk 清单，用来人工（或借助大模型）编写评测集。

    python -m scripts.export_chunks

只做加载和切分，不加载模型、不连接 Qdrant，所以 uvicorn 正在运行时也可以执行。
输出：data/evaluation/chunks_catalog.json
"""

import json
import logging
from pathlib import Path

from backend.core.config import settings
from backend.services.document_service import DocumentService


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("Enterprise RAG Assistant - Chunk 清单导出")
    print("=" * 60)

    # 这里只用到 loader 和 splitter，不需要 embedding 和向量库
    service = DocumentService(embedding_model=None, vector_store=None)

    documents = service.load_all_documents()
    chunks = service.splitter.split_documents(documents)

    if not chunks:
        print("错误：当前知识库没有任何 Chunk。")
        return

    export_data = [
        {
            "chunk_id": chunk.id,
            "content": chunk.content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]

    output_dir: Path = settings.project_root / "data" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "chunks_catalog.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    unique_ids = {item["chunk_id"] for item in export_data}
    empty_chunks = sum(1 for item in export_data if not item["content"].strip())

    print("\n导出完成！")
    print("-" * 40)
    print(f"原始文档对象：{len(documents)}")
    print(f"Chunk 总数：{len(export_data)}")
    print(f"唯一 ID 数量：{len(unique_ids)}")
    print(f"空 Chunk 数量：{empty_chunks}")
    print(f"保存位置：{output_file}")

    print("\n前 3 个 Chunk 预览：")
    for item in export_data[:3]:
        print("-" * 40)
        print(f"ID：{item['chunk_id']}")
        print(f"正文：{item['content'][:150]}")

    if len(unique_ids) != len(export_data):
        print("\n警告：发现重复 Chunk ID，请进一步检查。")


if __name__ == "__main__":
    main()
