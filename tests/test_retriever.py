from backend.rag.retriever import HybridRetriever, tokenize
from tests.helpers import FakeEmbedding, FakeReranker, FakeVectorStore, make_chunk


def build(chunks, store_chunks=None):
    """store_chunks 可以比 chunks 多，用来模拟向量库里有、本地索引里没有的记录。"""
    embedding = FakeEmbedding()
    store = FakeVectorStore()
    docs = store_chunks if store_chunks is not None else chunks

    if docs:
        store.add_documents(docs, embedding.encode_documents(docs))

    retriever = HybridRetriever(chunks, embedding, store, FakeReranker())
    return retriever


CHUNKS = [
    make_chunk("D1:chunk:0", "部署 Qdrant 时使用 Docker 启动服务并配置持久化目录"),
    make_chunk("D2:chunk:0", "PostgreSQL 用于存储用户信息和文档登记信息"),
    make_chunk("D3:chunk:0", "项目使用 Python 3.11.x 作为后端运行时"),
]


def test_tokenize_keeps_terms_and_drops_punctuation():
    tokens = tokenize("Python 3.11.x 的版本，.env / -")

    assert "python" in tokens
    assert any("3.11" in t for t in tokens)
    assert "." not in tokens and "/" not in tokens and "-" not in tokens
    assert all(t.strip() for t in tokens)


def test_empty_knowledge_base_returns_nothing():
    retriever = build([])

    assert retriever.bm25 is None
    assert retriever.retrieve("任何问题") == []


def test_bm25_ignores_zero_score_chunks():
    retriever = build(CHUNKS)

    assert retriever._bm25_search("zzzzqqqq", 5) == []
    assert retriever._bm25_search("Qdrant 部署", 5)[0] == "D1:chunk:0"


def test_hybrid_retrieval_puts_relevant_chunk_first():
    retriever = build(CHUNKS)

    results = retriever.retrieve("Qdrant 向量数据库如何部署", top_k=3, candidate_k=3)

    assert results[0]["chunk_id"] == "D1:chunk:0"
    assert "rerank_score" in results[0] and "rrf_score" in results[0]
    assert "bm25_rank" in results[0]["details"]


def test_chunk_found_by_both_lists_gets_higher_rrf_score():
    retriever = build(CHUNKS)

    results = retriever.retrieve("Qdrant 部署", top_k=3, candidate_k=3)
    by_id = {r["chunk_id"]: r for r in results}
    both = [r for r in results if {"vector_rank", "bm25_rank"} <= set(r["details"])]
    single = [r for r in results if len({"vector_rank", "bm25_rank"} & set(r["details"])) == 1]

    assert by_id["D1:chunk:0"]["details"].get("bm25_rank") == 1
    if both and single:
        assert max(r["rrf_score"] for r in both) > max(r["rrf_score"] for r in single)


def test_record_only_in_vector_store_is_still_returned():
    ghost = make_chunk("GHOST:chunk:0", "只存在于向量库里的内容")
    retriever = build(CHUNKS, store_chunks=CHUNKS + [ghost])

    # FakeEmbedding 对相同文本给出相同向量，所以这条查询会精确命中 ghost
    results = retriever.retrieve("只存在于向量库里的内容", top_k=5, candidate_k=5)

    assert "GHOST:chunk:0" in {r["chunk_id"] for r in results}


def test_empty_query_raises():
    retriever = build(CHUNKS)

    try:
        retriever.retrieve("   ")
    except ValueError:
        return

    raise AssertionError("空查询应该抛出 ValueError")
