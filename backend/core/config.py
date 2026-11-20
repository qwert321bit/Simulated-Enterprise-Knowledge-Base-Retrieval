"""集中管理路径、模型名称和各类阈值。其他模块只从这里读取，不再各自定义。"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class Settings:
    # ---------- 路径 ----------
    project_root: Path = PROJECT_ROOT
    knowledge_dir: Path = DATA_DIR / "enterprise"
    metadata_file: Path = DATA_DIR / "metadata" / "sources.json"
    upload_dir: Path = DATA_DIR / "uploads"
    qdrant_path: Path = DATA_DIR / "qdrant_storage"
    collection_name: str = "enterprise_knowledge"

    # ---------- 文件格式 ----------
    supported_extensions: frozenset = frozenset(
        {".md", ".txt", ".html", ".htm", ".pdf", ".docx"}
    )
    upload_extensions: frozenset = frozenset({".pdf", ".docx", ".txt", ".md"})
    max_upload_size: int = 10 * 1024 * 1024  # 10 MB
    upload_read_chunk: int = 1024 * 1024  # 每次读 1 MB

    # ---------- 切分 ----------
    chunk_size: int = 600
    chunk_overlap: int = 100

    # ---------- 模型 ----------
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "auto"
    embedding_batch_size: int = 8
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"
    reranker_max_length: int = 512

    # ---------- 检索 ----------
    default_top_k: int = 3
    candidate_k: int = 10
    rrf_k: int = 60
    # 低相关性拒答阈值：初始实验值，需要用 data/evaluation 的评测集校准
    min_rerank_score: float = 0.2

    # ---------- 生成 ----------
    max_context_chars: int = 12000
    deepseek_api_key: str | None = os.getenv("DEEPSEEK_API_KEY")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "60"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))

    def ensure_dirs(self) -> None:
        for path in (
            self.knowledge_dir,
            self.upload_dir,
            self.qdrant_path,
            self.metadata_file.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
