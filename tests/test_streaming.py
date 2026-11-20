"""流式输出：生成器、问答服务、SSE 编码。不依赖 FastAPI。"""

import json

import pytest

from backend.api.sse import encode_event, stream_events
from backend.core.exceptions import LLMError
from backend.rag.generator import NO_ANSWER, RAGGenerator
from backend.services.chat_service import LOW_RELEVANCE_ANSWER, ChatService
from tests.helpers import FakeLLM


def parse(events: list[str]) -> list[tuple[str, dict]]:
    parsed = []
    for frame in events:
        event_line, data_line = frame.strip().split("\n")
        parsed.append((event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))))
    return parsed


class CloseTracker:
    """带 close() 的迭代器，用来验证上游连接被关闭。"""

    def __init__(self, items, error=None):
        self.items, self.error, self.closed = list(items), error, False

    def __iter__(self):
        return self

    def __next__(self):
        if self.items:
            return self.items.pop(0)
        if self.error:
            raise self.error
        raise StopIteration

    def close(self):
        self.closed = True


# ----------------------------- SSE 编码 -----------------------------
def test_encode_event_is_single_frame_with_escaped_newlines():
    frame = encode_event("token", {"text": "第一行\n第二行"})

    assert frame == 'event: token\ndata: {"text": "第一行\\n第二行"}\n\n'
    assert frame.count("\n\n") == 1  # 文本里的换行被转义，不会提前结束这一帧


def test_event_order_is_sources_tokens_done():
    events = parse(list(stream_events({"query": "q", "sources": []}, "甲", iter(["乙", "丙"]))))

    assert [name for name, _ in events] == ["sources", "token", "token", "token", "done"]
    assert "".join(d["text"] for n, d in events if n == "token") == "甲乙丙"


def test_no_first_token_still_sends_sources_and_done():
    events = parse(list(stream_events({"query": "q"}, None, iter([]))))

    assert [name for name, _ in events] == ["sources", "done"]


def test_midstream_llm_error_becomes_error_event_without_details():
    tokens = CloseTracker(["乙"], error=LLMError("upstream secret detail"))
    frames = list(stream_events({"query": "q"}, "甲", tokens))
    events = parse(frames)

    assert [name for name, _ in events] == ["sources", "token", "token", "error"]
    assert "done" not in [name for name, _ in events]
    assert "secret" not in "".join(frames)
    assert tokens.closed


def test_unexpected_error_becomes_generic_error_event():
    frames = list(stream_events({"query": "q"}, "甲", CloseTracker([], error=RuntimeError("bug"))))

    assert parse(frames)[-1][0] == "error" and "bug" not in "".join(frames)


def test_closing_event_stream_closes_upstream():
    tokens = CloseTracker(["a", "b", "c"])
    events = stream_events({"query": "q"}, None, tokens)

    next(events)  # sources
    next(events)  # 第一个 token
    events.close()  # 客户端断开

    assert tokens.closed


# --------------------------- RAGGenerator.stream ---------------------------
def chunk(content="Python 3.11.x"):
    return {"chunk_id": "A:chunk:0", "content": content,
            "metadata": {"title": "文档A", "source": "a.md"}, "rerank_score": 0.9}


def test_generator_stream_returns_sources_before_text():
    llm = FakeLLM(tokens=["Py", "thon"])
    sources, tokens = RAGGenerator(llm=llm).stream("版本？", [chunk()])

    assert sources[0]["index"] == 1 and sources[0]["title"] == "文档A"
    assert llm.calls == []  # 还没开始迭代，不应该已经调用大模型
    assert "".join(tokens) == "Python"
    assert "<context>" in llm.calls[0]["user"]


def test_generator_stream_without_context_skips_llm():
    llm = FakeLLM()
    sources, tokens = RAGGenerator(llm=llm).stream("问题", [chunk("   ")])

    assert sources == [] and list(tokens) == [NO_ANSWER] and llm.calls == []


def test_generator_stream_rejects_blank_query():
    with pytest.raises(ValueError):
        RAGGenerator(llm=FakeLLM()).stream("  ", [chunk()])


# --------------------------- ChatService.ask_stream ---------------------------
class StubDocuments:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query, top_k, candidate_k):
        return self.results


class StubGenerator:
    def __init__(self):
        self.stream_calls = 0

    def stream(self, query, retrieved_chunks):
        self.stream_calls += 1
        return [{"index": 1}], iter(["流式", "答案"])


def test_ask_stream_empty_retrieval_streams_no_answer_message():
    generator = StubGenerator()
    result = ChatService(StubDocuments([]), generator).ask_stream("问题")

    assert list(result.tokens) == [NO_ANSWER]
    assert result.sources == [] and result.retrieved_count == 0 and generator.stream_calls == 0


def test_ask_stream_low_score_refuses_without_calling_llm():
    generator = StubGenerator()
    docs = StubDocuments([{"content": "x", "rerank_score": 0.05}])
    result = ChatService(docs, generator).ask_stream("问题")

    assert list(result.tokens) == [LOW_RELEVANCE_ANSWER]
    assert result.sources == [] and result.retrieved_count == 1 and generator.stream_calls == 0


def test_ask_stream_relevant_results_stream_from_generator():
    generator = StubGenerator()
    docs = StubDocuments([{"content": "x", "rerank_score": 0.9}, {"content": "y", "rerank_score": 0.5}])
    result = ChatService(docs, generator).ask_stream("  问题  ")

    assert result.query == "问题" and result.retrieved_count == 2
    assert result.sources == [{"index": 1}] and "".join(result.tokens) == "流式答案"


def test_ask_stream_blank_question_raises():
    with pytest.raises(ValueError):
        ChatService(StubDocuments([]), StubGenerator()).ask_stream("   ")
