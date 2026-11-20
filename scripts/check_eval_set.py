"""
检查评测集里的 gold_snippets 是否都能在当前 Chunk 清单里找到。
重新切分或修改文档后先跑它，再跑 evaluate。

    python -m scripts.export_chunks
    python -m scripts.check_eval_set [--eval-set data/evaluation/eval_set.jsonl]

不加载任何模型，随时可以运行。
"""

import argparse
import json
from pathlib import Path

from backend.core.config import settings

EVAL_DIR = settings.project_root / "data" / "evaluation"


def main() -> None:
    parser = argparse.ArgumentParser(description="检查评测集证据句覆盖情况")
    parser.add_argument("--eval-set", type=Path, default=EVAL_DIR / "eval_set.jsonl")
    parser.add_argument("--catalog", type=Path, default=EVAL_DIR / "chunks_catalog.json")
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    items = [
        json.loads(line)
        for line in args.eval_set.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    missing = generic = 0

    for item in items:
        if not item.get("answerable", True):
            continue

        for snippet in item.get("gold_snippets", []):
            hits = [c["chunk_id"] for c in catalog if snippet in c["content"]]

            if not hits:
                missing += 1
                print(f"[找不到] {item['id']}: {snippet}")
            elif len(hits) > 3:
                generic += 1
                print(f"[过于笼统，命中 {len(hits)} 个 Chunk] {item['id']}: {snippet}")

    answerable = sum(1 for i in items if i.get("answerable", True))
    print(
        f"\n共 {len(items)} 题（可回答 {answerable}），"
        f"找不到证据句 {missing} 处，过于笼统 {generic} 处"
    )

    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
