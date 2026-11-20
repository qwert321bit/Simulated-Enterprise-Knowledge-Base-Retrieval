"""文档加载：支持 md / txt / html / pdf / docx，统一成 Document 对象。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from docx import Document as WordDocument
from pypdf import PdfReader

from backend.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """统一的文档数据结构（原文档和 Chunk 都用它）。"""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "content": self.content, "metadata": self.metadata}


class DocumentLoader:
    """知识库文档加载器。"""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        metadata_file: str | Path | None = None,
    ):
        self.data_dir = Path(data_dir or settings.knowledge_dir)
        self.metadata_file = Path(metadata_file or settings.metadata_file)
        self.source_registry = self._load_registry()

    # ------------------------------------------------------------------
    # 来源登记表
    # ------------------------------------------------------------------
    def _load_registry(self) -> dict[str, dict]:
        if not self.metadata_file.exists():
            logger.warning("来源登记表不存在：%s", self.metadata_file)
            return {}

        try:
            with self.metadata_file.open("r", encoding="utf-8") as f:
                records = json.load(f)

            return {
                item["file_name"]: item
                for item in records
                if item.get("file_name")
            }
        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.error("读取来源登记表失败：%s", e)
            return {}

    # ------------------------------------------------------------------
    # 各格式解析
    # ------------------------------------------------------------------
    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            # 兼容 Windows 下的 GBK 编码文本
            return path.read_text(encoding="gb18030")

    def _read_html(self, path: Path) -> str:
        soup = BeautifulSoup(self._read_text(path), "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        for heading in soup.find_all(["h1", "h2", "h3", "h4"]):
            heading.insert_before("\n\n")
            heading.append("\n")

        return soup.get_text(separator="\n", strip=True)

    @staticmethod
    def _read_pdf(path: Path) -> list[tuple[str, int]]:
        """每页独立返回，保留页码，方便引用来源。"""
        reader = PdfReader(str(path))
        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((text, page_number))

        return pages

    @staticmethod
    def _read_docx(path: Path) -> str:
        doc = WordDocument(str(path))
        parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))

        return "\n".join(parts)

    def _parse_file(self, path: Path) -> list[tuple[str, int | None]]:
        suffix = path.suffix.lower()

        if suffix in {".md", ".txt"}:
            return [(self._read_text(path), None)]
        if suffix in {".html", ".htm"}:
            return [(self._read_html(path), None)]
        if suffix == ".pdf":
            return self._read_pdf(path)
        if suffix == ".docx":
            return [(self._read_docx(path), None)]

        return []

    # ------------------------------------------------------------------
    # 元数据
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_title(content: str, name: str) -> str:
        """优先取 Markdown 一级标题，否则用文件名（不含扩展名）。"""
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()

        return Path(name).stem

    def _build_metadata(
        self,
        path: Path,
        content: str,
        page_number: int | None,
        display_name: str | None,
        document_id: str | None,
    ) -> dict[str, Any]:
        # 上传文件在磁盘上的名字可能带前缀，对外一律使用 display_name
        filename = display_name or path.name
        registered = self.source_registry.get(filename, {})

        doc_id = document_id or registered.get("document_id", path.stem)

        source = registered.get("source_url")
        if not source:
            if display_name:
                source = display_name
            else:
                try:
                    source = str(path.relative_to(settings.project_root))
                except ValueError:
                    source = str(path)

        metadata = {
            "document_id": doc_id,
            "title": self._extract_title(content, filename),
            "file_name": filename,
            "file_path": str(path),
            "source": source,
            "category": registered.get("category", "uncategorized"),
            "language": registered.get("language", "unknown"),
            "document_type": registered.get("document_type", "unknown"),
            "source_type": registered.get("source_type", "unknown"),
            "license": registered.get("license", "unknown"),
            "version": registered.get("version", "unknown"),
        }

        if page_number is not None:
            metadata["page_number"] = page_number

        return metadata

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def load_file(
        self,
        path: str | Path,
        display_name: str | None = None,
        document_id: str | None = None,
    ) -> list[Document]:
        """
        加载单个文件。解析失败会直接抛出异常，由调用方决定如何处理。

        display_name: 对外显示的文件名（上传文件用原始文件名）
        document_id:  指定文档 ID（上传文件用 upload:<upload_id>，避免重名冲突）
        """
        path = Path(path)
        documents = []

        for content, page_number in self._parse_file(path):
            if not content.strip():
                logger.warning("文档内容为空，跳过：%s", path)
                continue

            metadata = self._build_metadata(
                path, content, page_number, display_name, document_id
            )

            base_id = metadata["document_id"]
            unique_id = (
                f"{base_id}:page:{page_number}"
                if page_number is not None
                else base_id
            )

            documents.append(
                Document(id=unique_id, content=content, metadata=metadata)
            )

        return documents

    def load_documents(self) -> list[Document]:
        """扫描 data_dir 并加载所有支持的文档；目录不存在时返回空列表。"""

        if not self.data_dir.exists():
            logger.warning("知识库目录不存在：%s", self.data_dir)
            return []

        documents: list[Document] = []

        for path in sorted(self.data_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in settings.supported_extensions:
                continue

            try:
                documents.extend(self.load_file(path))
                logger.info("加载成功：%s", path.name)
            except Exception:
                logger.exception("加载文档失败：%s", path)

        logger.info("文档加载完成，共 %d 个文档对象", len(documents))
        return documents
