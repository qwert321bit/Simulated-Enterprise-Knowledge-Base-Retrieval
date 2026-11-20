"""HTTP 层：状态码映射、上传校验、删除。用 TestClient，不启动 lifespan，不加载模型。"""

import json
import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api import chat, documents
from backend.core.exceptions import LLMError


def make_client(chat_service=None, document_service=None) -> TestClient:
    app = FastAPI()
    app.include_router(chat.router)
    app.include_router(documents.router)

    if chat_service is not None:
        app.state.chat_service = chat_service
    if document_service is not None:
        app.state.document_service = document_service

    return TestClient(app)


class FakeChatService:
    def __init__(self, error=None, tokens=None, midstream_error=None):
        self.error = error
        self.tokens = tokens if tokens is not None else ["答案", " [1]"]
        self.midstream_error = midstream_error

    def _token_iter(self):
        yield from self.tokens
        if self.midstream_error:
            raise self.midstream_error

    def ask_stream(self, query, top_k, candidate_k):
        if self.error:
            raise self.error
        if not query.strip():
            raise ValueError("问题不能为空")
        return SimpleNamespace(
            query=query,
            sources=[{"index": 1, "title": "T", "source": "a.md", "heading": "H",
                      "page_number": None, "chunk_id": "A:chunk:0", "rerank_score": 0.9}],
            retrieved_count=2,
            tokens=self._token_iter(),
        )

    def ask(self, query, top_k, candidate_k):
        if self.error:
            raise self.error
        if not query.strip():
            raise ValueError("问题不能为空")
        return {
            "query": query,
            "answer": "答案 [1]",
            "sources": [{"index": 1, "title": "T", "source": "a.md", "heading": "H",
                         "page_number": None, "chunk_id": "A:chunk:0", "rerank_score": 0.9}],
            "retrieved_chunks": [{}, {}],
        }


# ---------------------------- /chat ----------------------------
def test_chat_success_matches_schema():
    response = make_client(chat_service=FakeChatService()).post("/chat", json={"query": "问题", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "答案 [1]" and body["retrieved_count"] == 2
    assert body["sources"][0]["title"] == "T"


@pytest.mark.parametrize("payload", [{"query": ""}, {"query": "x", "top_k": 0}, {"query": "x", "top_k": 11}])
def test_chat_validates_request(payload):
    assert make_client(chat_service=FakeChatService()).post("/chat", json=payload).status_code == 422


def test_chat_blank_question_is_400():
    assert make_client(chat_service=FakeChatService()).post("/chat", json={"query": "   "}).status_code == 400


def test_chat_llm_failure_is_502():
    client = make_client(chat_service=FakeChatService(error=LLMError("boom")))
    response = client.post("/chat", json={"query": "问题"})

    assert response.status_code == 502
    assert "boom" not in response.text  # 不向客户端泄露上游错误细节


def test_chat_unexpected_error_is_500():
    client = make_client(chat_service=FakeChatService(error=RuntimeError("secret detail")))
    response = client.post("/chat", json={"query": "问题"})

    assert response.status_code == 500 and "secret detail" not in response.text


def test_chat_before_initialisation_is_503():
    assert make_client().post("/chat", json={"query": "问题"}).status_code == 503


# ------------------------- /chat/stream -------------------------
def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for frame in text.strip().split("\n\n"):
        event_line, data_line = frame.split("\n")
        events.append((event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))))
    return events


def test_stream_success_event_sequence():
    client = make_client(chat_service=FakeChatService(tokens=["答", "案", "[1]"]))
    response = client.post("/chat/stream", json={"query": "问题", "top_k": 2})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["sources", "token", "token", "token", "done"]
    assert events[0][1]["retrieved_count"] == 2 and events[0][1]["sources"][0]["title"] == "T"
    assert "".join(d["text"] for n, d in events if n == "token") == "答案[1]"


@pytest.mark.parametrize("payload", [{"query": ""}, {"query": "x", "top_k": 11}])
def test_stream_validates_request(payload):
    assert make_client(chat_service=FakeChatService()).post("/chat/stream", json=payload).status_code == 422


def test_stream_blank_question_is_400():
    assert make_client(chat_service=FakeChatService()).post("/chat/stream", json={"query": "   "}).status_code == 400


def test_stream_before_initialisation_is_503():
    assert make_client().post("/chat/stream", json={"query": "问题"}).status_code == 503


def test_stream_failure_before_first_token_is_502():
    # 鉴权失败、连接超时等错误发生在第一段文本之前，应该是正常的 HTTP 错误码
    client = make_client(chat_service=FakeChatService(tokens=[], midstream_error=LLMError("boom")))
    response = client.post("/chat/stream", json={"query": "问题"})

    assert response.status_code == 502 and "boom" not in response.text


def test_stream_failure_midway_is_an_error_event():
    client = make_client(chat_service=FakeChatService(tokens=["前半"], midstream_error=LLMError("boom")))
    response = client.post("/chat/stream", json={"query": "问题"})
    events = parse_sse(response.text)

    assert response.status_code == 200
    assert [name for name, _ in events] == ["sources", "token", "error"]
    assert "boom" not in response.text


# ------------------------- /documents -------------------------
def upload(client, name="notes.txt", content="星云科技测试服务器端口为 8765。".encode()):
    return client.post("/documents/upload", files={"file": (name, content, "text/plain")})


def test_upload_success_keeps_original_filename(doc_service):
    response = upload(make_client(document_service=doc_service))

    assert response.status_code == 200
    body = response.json()
    assert body["success"] and body["file_name"] == "notes.txt" and body["chunk_count"] >= 1
    assert len(body["upload_id"]) == 32


def test_upload_rejects_unsupported_extension(doc_service):
    assert upload(make_client(document_service=doc_service), name="run.exe").status_code == 400


def test_upload_rejects_empty_file(doc_service):
    assert upload(make_client(document_service=doc_service), content=b"").status_code == 400


def test_upload_rejects_oversized_file(doc_service, override_settings):
    override_settings(max_upload_size=10)
    assert upload(make_client(document_service=doc_service), content=b"x" * 100).status_code == 413


def test_upload_strips_path_components_from_filename(doc_service):
    response = upload(make_client(document_service=doc_service), name="../../evil.txt")

    assert response.status_code == 200 and response.json()["file_name"] == "evil.txt"


def test_duplicate_upload_is_409_and_cleans_up(doc_service, patched_settings):
    client = make_client(document_service=doc_service)
    assert upload(client).status_code == 200

    before = len(list(patched_settings.upload_dir.iterdir()))
    assert upload(client).status_code == 409
    assert len(list(patched_settings.upload_dir.iterdir())) == before


def test_unparseable_file_is_422(doc_service):
    assert upload(make_client(document_service=doc_service), name="bad.docx", content=b"not a zip").status_code == 422


def test_list_and_delete_document(doc_service):
    client = make_client(document_service=doc_service)
    upload_id = upload(client).json()["upload_id"]

    listing = client.get("/documents").json()
    assert listing["total"] == 1 and listing["documents"][0]["deletable"] is True

    deleted = client.delete(f"/documents/{upload_id}")
    assert deleted.status_code == 200 and deleted.json()["deleted_chunks"] >= 1
    assert client.get("/documents").json()["total"] == 0


def test_delete_unknown_document_is_404(doc_service):
    client = make_client(document_service=doc_service)

    assert client.delete(f"/documents/{uuid.uuid4().hex}").status_code == 404
    assert client.delete("/documents/not-a-valid-id").status_code == 404
