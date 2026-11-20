"""切分、生成、问答流程、文档入库与删除。"""

import uuid

import pytest

from backend.core.exceptions import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    UnprocessableDocumentError,
)
from backend.rag.generator import NO_ANSWER, RAGGenerator
from backend.rag.loader import Document
from backend.rag.splitter import TextSplitter
from backend.services.chat_service import LOW_RELEVANCE_ANSWER, ChatService
from tests.helpers import FakeEmbedding, FakeLLM, FakeVectorStore, make_chunk, save_upload


# ----------------------------------------------------------------------
# TextSplitter
# ----------------------------------------------------------------------
def split(text, size=300, overlap=60, title="手册"):
    return TextSplitter(size, overlap).split_document(Document("T", text, {"title": title}))


def test_invalid_splitter_params():
    with pytest.raises(ValueError):
        TextSplitter(chunk_size=0)
    with pytest.raises(ValueError):
        TextSplitter(chunk_size=100, chunk_overlap=100)


def test_chunk_carries_title_and_heading_context():
    chunks = split("# 手册\n\n## 2. 技术基线\n\nPython 版本为 3.11.x。")

    assert chunks[0].content.startswith("文档标题：手册")
    assert "所属章节：2. 技术基线" in chunks[0].content
    assert chunks[0].metadata["heading"] == "2. 技术基线"
    assert chunks[0].id == "T:chunk:0"


def test_version_numbers_and_filenames_are_not_split():
    text = "Install Python 3.11.x first. Then copy .env.example to .env. " * 30
    joined = "\n".join(c.content for c in split(text, size=200, overlap=40))

    assert "3.11.x" in joined and ".env.example" in joined
    assert "3.\n11" not in joined


def test_long_paragraph_chunks_overlap():
    text = "这是一个用来测试重叠切分的句子。" * 80
    chunks = split(text, size=200, overlap=50)

    assert len(chunks) > 2
    first, second = (c.content.split("\n\n", 1)[1] for c in chunks[:2])
    assert first[-15:] in second[:120]


def test_code_block_is_split_by_lines_with_fences():
    lines = [f"  key_{i}: value_{i}" for i in range(60)]
    text = "## 配置\n\n```yaml\n" + "\n".join(lines) + "\n```\n"
    chunks = split(text, size=250, overlap=40)

    assert len(chunks) > 1
    for chunk in chunks:
        body = chunk.content.split("\n\n", 1)[1]
        assert body.startswith("```yaml") and body.rstrip().endswith("```")

    joined = "\n".join(c.content for c in chunks)
    assert all(line in joined for line in lines)


def test_hash_comment_in_code_block_is_not_a_heading():
    chunks = split("## 命令\n\n```bash\n# 这是注释\npython -m venv .venv\n```\n")

    assert len(chunks) == 1
    assert "# 这是注释" in chunks[0].content
    assert chunks[0].metadata["heading"] == "命令"


def test_long_table_repeats_header_row():
    rows = "\n".join(f"| 组件{i} | {i}.0 |" for i in range(80))
    chunks = split("## 表\n\n| 组件 | 版本 |\n|---|---|\n" + rows + "\n", size=300, overlap=50)

    assert len(chunks) > 1
    assert all("| 组件 | 版本 |" in c.content for c in chunks)


# ----------------------------------------------------------------------
# RAGGenerator
# ----------------------------------------------------------------------
def chunk_dict(content, **extra):
    return {
        "chunk_id": extra.pop("chunk_id", "A:chunk:0"),
        "content": content,
        "metadata": {"title": "文档A", "source": "a.md", **extra},
        "rerank_score": 0.9,
    }


def test_build_context_numbers_skip_empty_chunks():
    generator = RAGGenerator(llm=FakeLLM())
    context, sources = generator.build_context(
        [chunk_dict("第一段"), chunk_dict("   "), chunk_dict("第三段", chunk_id="B:chunk:0")]
    )

    assert [s["index"] for s in sources] == [1, 2]
    assert "[1] 文档：文档A" in context and "[2] 文档：文档A" in context
    assert "[3]" not in context


def test_build_context_respects_char_limit():
    generator = RAGGenerator(llm=FakeLLM())
    context, sources = generator.build_context(
        [chunk_dict("甲" * 500), chunk_dict("乙" * 500)], max_context_chars=300
    )

    assert len(context) <= 300 + 2  # 块之间的分隔
    assert len(sources) == 1


def test_generate_without_chunks_skips_llm():
    llm = FakeLLM()
    result = RAGGenerator(llm=llm).generate("问题", [])

    assert result["answer"] == NO_ANSWER and result["sources"] == []
    assert llm.calls == []


def test_generate_sends_numbered_context_to_llm():
    llm = FakeLLM(answer="答案 [1]")
    result = RAGGenerator(llm=llm).generate("Python 版本？", [chunk_dict("Python 3.11.x")])

    assert result["answer"] == "答案 [1]" and result["model"] == "fake-model"
    assert "<context>" in llm.calls[0]["user"] and "[1]" in llm.calls[0]["user"]
    assert "Python 版本？" in llm.calls[0]["user"]


def test_generate_rejects_blank_query():
    with pytest.raises(ValueError):
        RAGGenerator(llm=FakeLLM()).generate("  ", [chunk_dict("x")])


# ----------------------------------------------------------------------
# ChatService
# ----------------------------------------------------------------------
class StubDocuments:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, query, top_k, candidate_k):
        self.calls.append((query, top_k, candidate_k))
        return self.results


class StubGenerator:
    def __init__(self):
        self.calls = 0

    def generate(self, query, retrieved_chunks):
        self.calls += 1
        return {"answer": "生成的答案", "sources": [{"index": 1}], "model": "m"}


def test_no_retrieval_results_returns_no_answer():
    generator = StubGenerator()
    result = ChatService(StubDocuments([]), generator).ask("问题")

    assert result["answer"] == NO_ANSWER and generator.calls == 0


def test_low_rerank_score_refuses_without_calling_llm():
    generator = StubGenerator()
    docs = StubDocuments([{"content": "x", "rerank_score": 0.05}])
    result = ChatService(docs, generator).ask("问题")

    assert result["answer"] == LOW_RELEVANCE_ANSWER
    assert result["sources"] == [] and generator.calls == 0


def test_relevant_results_go_to_generator():
    generator = StubGenerator()
    docs = StubDocuments([{"content": "x", "rerank_score": 0.9}])
    result = ChatService(docs, generator).ask("  问题  ", top_k=5, candidate_k=3)

    assert result["answer"] == "生成的答案" and generator.calls == 1
    assert docs.calls == [("问题", 5, 5)]  # candidate_k 不会小于 top_k


def test_blank_question_raises():
    with pytest.raises(ValueError):
        ChatService(StubDocuments([]), StubGenerator()).ask("   ")


# ----------------------------------------------------------------------
# DocumentService
# ----------------------------------------------------------------------
def test_startup_only_embeds_missing_chunks(patched_settings):
    from backend.services.document_service import DocumentService

    (patched_settings.knowledge_dir / "a.md").write_text("# 文档A\n\n内容甲。", encoding="utf-8")
    store, embedding = FakeVectorStore(), FakeEmbedding()

    DocumentService(embedding, store).load_knowledge_base()
    assert store.count() >= 1 and embedding.document_calls == 1

    DocumentService(embedding, store).load_knowledge_base()  # 向量库已是最新
    assert embedding.document_calls == 1

    (patched_settings.knowledge_dir / "b.md").write_text("# 文档B\n\n内容乙。", encoding="utf-8")
    before = store.count()
    DocumentService(embedding, store).load_knowledge_base()
    assert store.count() > before and embedding.document_calls == 2


def test_add_document_uses_original_name_not_uuid(doc_service):
    upload_id, path = save_upload("规范.txt", "星云科技测试服务器端口为 8765。".encode())

    result = doc_service.add_document(path, "规范.txt", upload_id)
    info = next(d for d in doc_service.list_documents() if d["upload_id"] == upload_id)

    assert result["file_name"] == "规范.txt" and result["chunk_count"] >= 1
    assert info["title"] == "规范" and info["file_name"] == "规范.txt"
    assert upload_id not in info["title"]


def test_duplicate_content_is_rejected(doc_service):
    id1, p1 = save_upload("a.txt", "相同的内容".encode())
    id2, p2 = save_upload("b.txt", "相同的内容".encode())

    doc_service.add_document(p1, "a.txt", id1)

    with pytest.raises(DuplicateDocumentError):
        doc_service.add_document(p2, "b.txt", id2)


def test_same_filename_different_content_do_not_collide(doc_service):
    id1, p1 = save_upload("a.txt", "第一份文档的内容".encode())
    id2, p2 = save_upload("a.txt", "第二份完全不同的文档内容".encode())

    first = doc_service.add_document(p1, "a.txt", id1)
    second = doc_service.add_document(p2, "a.txt", id2)

    assert doc_service.vector_store.count() == first["chunk_count"] + second["chunk_count"]


def test_unparseable_and_empty_files_are_rejected_and_not_remembered(doc_service):
    id1, blank = save_upload("blank.txt", b"   \n  ")

    with pytest.raises(UnprocessableDocumentError):
        doc_service.add_document(blank, "blank.txt", id1)

    # 失败后去重记录必须释放：再次提交仍然是"无法解析"，而不是"重复"
    id2, again = save_upload("blank2.txt", b"   \n  ")
    with pytest.raises(UnprocessableDocumentError):
        doc_service.add_document(again, "blank2.txt", id2)

    id3, broken = save_upload("bad.docx", b"this is not a zip file")
    with pytest.raises(UnprocessableDocumentError):
        doc_service.add_document(broken, "bad.docx", id3)


def test_delete_document_cleans_everything(doc_service, patched_settings):
    upload_id, path = save_upload("a.txt", "要被删除的文档".encode())
    doc_service.add_document(path, "a.txt", upload_id)
    assert doc_service.vector_store.count() > 0

    result = doc_service.delete_document(upload_id)

    assert result["deleted_chunks"] >= 1 and result["vector_count"] == 0
    assert doc_service.list_documents() == []
    assert not (patched_settings.upload_dir / upload_id).exists()

    # 去重记录已释放，同一份内容可以重新上传
    new_id, new_path = save_upload("a.txt", "要被删除的文档".encode())
    doc_service.add_document(new_path, "a.txt", new_id)


def test_delete_unknown_or_malformed_id(doc_service):
    with pytest.raises(DocumentNotFoundError):
        doc_service.delete_document(uuid.uuid4().hex)

    with pytest.raises(DocumentNotFoundError):
        doc_service.delete_document("../../etc")


def test_knowledge_base_documents_are_not_deletable(patched_settings):
    from backend.services.document_service import DocumentService

    (patched_settings.knowledge_dir / "E001.md").write_text("# 手册\n\n正文。", encoding="utf-8")
    service = DocumentService(FakeEmbedding(), FakeVectorStore())
    service.load_knowledge_base()

    docs = service.list_documents()
    assert docs and all(not d["deletable"] and d["upload_id"] is None for d in docs)
