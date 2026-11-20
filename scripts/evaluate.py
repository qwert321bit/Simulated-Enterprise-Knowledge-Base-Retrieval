"""
检索评测：Hit@K、MRR、消融对比（向量 / BM25 / RRF / +Reranker）、拒答阈值校准。
只评测检索，不调用大模型，所以不花 API 费用。

    python -m scripts.evaluate
    python -m scripts.evaluate --eval-set data/evaluation/eval_set.jsonl

注意：Qdrant 本地模式是文件锁，运行前先停掉 uvicorn。

评测集是 JSONL，每行一道题：
{"id": "q1", "question": "...", "answerable": true,
 "gold_chunk_ids": ["E001:chunk:2"], "gold_snippets": ["Python 3.11"]}

- answerable=false 的题表示知识库里没有答案，只用于校准拒答阈值，不需要 gold。
- 命中规则：检索到的 Chunk 的 ID 在 gold_chunk_ids 里，或正文包含任一 gold_snippets。
- 推荐用 gold_snippets：重新切分后 Chunk ID 会变，关键句不会。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

from backend.core.config import settings
from backend.rag.embedding import EmbeddingModel
from backend.rag.reranker import Reranker
from backend.rag.vector_store import QdrantVectorStore
from backend.services.document_service import DocumentService

EVAL_DIR = settings.project_root / "data" / "evaluation"
DEFAULT_EVAL_SET = EVAL_DIR / "eval_set.jsonl"
DEFAULT_REPORT = EVAL_DIR / "eval_report.json"

KS = (1, 3, 5)
# 取大一点的候选数，保证消融时向量/BM25 各自的前 5 名都还在候选里
CANDIDATE_K = 20
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
VIEWS = ("向量检索", "BM25", "RRF 混合", "混合 + Reranker")


# ----------------------------------------------------------------------
# 评测集
# ----------------------------------------------------------------------
def load_eval_set(path: Path) -> list[dict[str, Any]]:
    items = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            if not item.get("question"):
                raise ValueError(f"第 {line_no} 行缺少 question")

            item.setdefault("id", f"q{line_no}")
            item.setdefault("answerable", True)
            item["gold_chunk_ids"] = item.get("gold_chunk_ids") or []
            item["gold_snippets"] = item.get("gold_snippets") or []

            if item["answerable"] and not (item["gold_chunk_ids"] or item["gold_snippets"]):
                raise ValueError(
                    f"第 {line_no} 行：answerable=true 的题必须有 gold_chunk_ids 或 gold_snippets"
                )

            items.append(item)

    return items


def is_hit(chunk: dict[str, Any], item: dict[str, Any]) -> bool:
    if chunk["chunk_id"] in item["gold_chunk_ids"]:
        return True
    content = chunk.get("content") or ""
    return any(snippet in content for snippet in item["gold_snippets"])


def first_hit_rank(ranked: list[dict[str, Any]], item: dict[str, Any]) -> int | None:
    for rank, chunk in enumerate(ranked, start=1):
        if is_hit(chunk, item):
            return rank
    return None


# ----------------------------------------------------------------------
# 同一次检索结果，按不同阶段的排序方式取出四份榜单
# ----------------------------------------------------------------------
def build_views(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "向量检索": sorted(
            (r for r in results if "vector_rank" in r["details"]),
            key=lambda r: r["details"]["vector_rank"],
        ),
        "BM25": sorted(
            (r for r in results if "bm25_rank" in r["details"]),
            key=lambda r: r["details"]["bm25_rank"],
        ),
        "RRF 混合": sorted(results, key=lambda r: r["rrf_score"], reverse=True),
        "混合 + Reranker": results,
    }


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def evaluate(service: DocumentService, items: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [i for i in items if i["answerable"]]
    unanswerable = [i for i in items if not i["answerable"]]

    hits = {v: {k: 0 for k in KS} for v in VIEWS}
    reciprocal = {v: 0.0 for v in VIEWS}
    answerable_scores: list[float] = []
    unanswerable_scores: list[float] = []
    unanswerable_by_type: dict[str, list[float]] = {}
    per_question: list[dict[str, Any]] = []

    started = time.time()

    for position, item in enumerate(items, start=1):
        t0 = time.time()
        results = service.retrieve(item["question"], CANDIDATE_K, CANDIDATE_K)
        elapsed = time.time() - t0
        done = time.time() - started
        eta = done / position * (len(items) - position)
        print(
            f"[{position}/{len(items)}] {item['id']}  本题 {elapsed:.1f}s，"
            f"已用 {done / 60:.1f} 分钟，预计还需 {eta / 60:.1f} 分钟",
            flush=True,
        )
        best_score = results[0]["rerank_score"] if results else 0.0
        record: dict[str, Any] = {
            "id": item["id"],
            "question": item["question"],
            "answerable": item["answerable"],
            "best_rerank_score": best_score,
            "top3": [
                {"chunk_id": r["chunk_id"], "rerank_score": round(r["rerank_score"], 4)}
                for r in results[:3]
            ],
        }

        if not item["answerable"]:
            unanswerable_scores.append(best_score)
            unanswerable_by_type.setdefault(
                item.get("negative_type", "unspecified"), []
            ).append(best_score)
            per_question.append(record)
            continue

        answerable_scores.append(best_score)
        views = build_views(results)
        ranks = {}

        for view, ranked in views.items():
            rank = first_hit_rank(ranked, item)
            ranks[view] = rank

            if rank is not None:
                reciprocal[view] += 1 / rank
                for k in KS:
                    if rank <= k:
                        hits[view][k] += 1

        record["first_hit_rank"] = ranks
        per_question.append(record)

    n = len(answerable)
    summary: dict[str, Any] = {
        "answerable_count": n,
        "unanswerable_count": len(unanswerable),
        "metrics": {
            view: {
                **{f"Hit@{k}": (hits[view][k] / n if n else 0.0) for k in KS},
                "MRR": (reciprocal[view] / n if n else 0.0),
            }
            for view in VIEWS
        },
        "threshold_sweep": [
            {
                "threshold": t,
                "answer_rate_on_answerable": (
                    sum(s >= t for s in answerable_scores) / len(answerable_scores)
                    if answerable_scores else None
                ),
                "refuse_rate_on_unanswerable": (
                    sum(s < t for s in unanswerable_scores) / len(unanswerable_scores)
                    if unanswerable_scores else None
                ),
            }
            for t in THRESHOLDS
        ],
        # 不可回答的题按 negative_type 分开看：near_miss（话题相关但资料里没有）
        # 比 off_topic（完全无关）难得多，混在一起会掩盖问题
        "refusal_by_type": {
            ntype: {
                str(t): sum(s < t for s in scores) / len(scores)
                for t in THRESHOLDS
            }
            for ntype, scores in unanswerable_by_type.items()
        },
    }

    return {"summary": summary, "questions": per_question}


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]

    print("\n" + "=" * 60)
    print(
        f"评测题数：可回答 {summary['answerable_count']}，"
        f"不可回答 {summary['unanswerable_count']}"
    )
    print("=" * 60)

    print("\n检索效果（只统计可回答的题）\n")
    print("| 方案 | Hit@1 | Hit@3 | Hit@5 | MRR |")
    print("|---|---|---|---|---|")
    for view, m in summary["metrics"].items():
        print(
            f"| {view} | {pct(m['Hit@1'])} | {pct(m['Hit@3'])} | "
            f"{pct(m['Hit@5'])} | {m['MRR']:.3f} |"
        )

    print("\n拒答阈值扫描（最高 rerank 分数 < 阈值就拒答）\n")
    print("| 阈值 | 可回答题仍作答 | 不可回答题正确拒答 |")
    print("|---|---|---|")
    for row in summary["threshold_sweep"]:
        print(
            f"| {row['threshold']} | {pct(row['answer_rate_on_answerable'])} | "
            f"{pct(row['refuse_rate_on_unanswerable'])} |"
        )

    by_type = summary.get("refusal_by_type", {})
    if len(by_type) > 1 or (by_type and "unspecified" not in by_type):
        print("\n不可回答题按类型的正确拒答率\n")
        header = "| 类型 | " + " | ".join(str(t) for t in THRESHOLDS) + " |"
        print(header)
        print("|---|" + "---|" * len(THRESHOLDS))
        for ntype, rates in by_type.items():
            print(f"| {ntype} | " + " | ".join(pct(rates[str(t)]) for t in THRESHOLDS) + " |")

    misses = [
        q for q in report["questions"]
        if q["answerable"]
        and (q["first_hit_rank"]["混合 + Reranker"] is None
             or q["first_hit_rank"]["混合 + Reranker"] > 5)
    ]

    if misses:
        print(f"\n最终排序前 5 名没命中的题（{len(misses)} 道）：")
        for q in misses:
            print(f"- [{q['id']}] {q['question']}")
            for r in q["top3"]:
                print(f"    {r['chunk_id']}  rerank={r['rerank_score']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索评测")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--sample", type=int, default=0,
        help="只评测约 N 题（均匀抽样，可回答和不可回答题都会抽到），用于快速试跑",
    )
    args = parser.parse_args()

    if not args.eval_set.exists():
        print(f"找不到评测集：{args.eval_set}")
        print("先运行 python -m scripts.export_chunks 导出 Chunk 清单，")
        print("再参照 data/evaluation/eval_set.example.jsonl 编写 eval_set.jsonl。")
        raise SystemExit(1)

    logging.basicConfig(level=logging.WARNING)
    settings.ensure_dirs()

    items = load_eval_set(args.eval_set)

    if args.sample and args.sample < len(items):
        step = math.ceil(len(items) / args.sample)
        items = items[::step]
        print(f"抽样评测：{len(items)} 题（每 {step} 题取 1 题）")

    embedding_model = EmbeddingModel()
    vector_store = QdrantVectorStore(dimension=embedding_model.dimension)

    try:
        service = DocumentService(embedding_model, vector_store, Reranker())
        service.load_knowledge_base()

        report = evaluate(service, items)
    finally:
        vector_store.close()

    print_report(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存：{args.output}")


if __name__ == "__main__":
    main()
