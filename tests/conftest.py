import pytest

from backend.core.config import settings
from backend.services.document_service import DocumentService
from tests.helpers import FakeEmbedding, FakeVectorStore


@pytest.fixture
def override_settings():
    """Settings 是 frozen dataclass，测试里临时改值并在结束后还原。"""
    saved = {}

    def _set(**kwargs):
        for key, value in kwargs.items():
            saved.setdefault(key, getattr(settings, key))
            object.__setattr__(settings, key, value)

    yield _set

    for key, value in saved.items():
        object.__setattr__(settings, key, value)


@pytest.fixture
def patched_settings(tmp_path, override_settings):
    """把所有数据目录指到临时目录，测试不会碰真实的 data/。"""
    paths = {
        "knowledge_dir": tmp_path / "enterprise",
        "upload_dir": tmp_path / "uploads",
        "qdrant_path": tmp_path / "qdrant",
        "metadata_file": tmp_path / "metadata" / "sources.json",
    }
    override_settings(**paths)

    for key, path in paths.items():
        (path.parent if key == "metadata_file" else path).mkdir(parents=True, exist_ok=True)

    return settings


@pytest.fixture
def doc_service(patched_settings):
    """只带假 Embedding 和假向量库的 DocumentService（不建检索器）。"""
    service = DocumentService(FakeEmbedding(), FakeVectorStore(), reranker=None)
    service.load_knowledge_base()
    return service
