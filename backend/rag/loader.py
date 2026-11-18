
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from bs4 import BeautifulSoup
from pypdf import PdfReader
from docx import Document as WordDocument


logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 默认知识库目录
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "enterprise"

# 来源元数据文件
DEFAULT_METADATA_FILE = PROJECT_ROOT / "data" / "metadata" / "sources.json"

# 当前支持的文件格式
SUPPORTED_EXTENSIONS = {
    ".md",
    ".txt",
    ".html",
    ".htm",
    ".pdf",
    ".docx",
}


@dataclass
class Document:
    """
    统一的文档数据结构
    """

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换成普通字典，方便调试或序列化。"""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
        }


class DocumentLoader:
    """知识库文档加载器。"""

    def __init__(
        self,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        metadata_file: str | Path = DEFAULT_METADATA_FILE,
    ):
        self.data_dir = Path(data_dir)
        self.metadata_file = Path(metadata_file)

        # 加载文档来源登记表
        self.source_registry = self._load_registry()

    def _load_registry(self) -> dict[str, dict]:
        """读取 sources.json，建立文件名到元数据的映射。"""

        if not self.metadata_file.exists():
            logger.warning(
                "来源登记表不存在：%s", self.metadata_file
            )
            return {}

        try:
            with self.metadata_file.open(
                "r", encoding="utf-8"
            ) as f:
                records = json.load(f)

            return {
                item["file_name"]: item
                for item in records
                if item.get("file_name")
            }

        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.error("读取来源登记表失败：%s", e)
            return {}

    def _read_text(self, path: Path) -> str:
        """读取普通文本、Markdown 文件。"""

        try:
            return path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            # 兼容部分 Windows 编码的文本文件
            return path.read_text(encoding="gb18030")

    def _read_html(self, path: Path) -> str:
        """提取 HTML 正文，去除导航栏、脚本和样式。"""

        html = self._read_text(path)
        soup = BeautifulSoup(html, "html.parser")

        # 删除无关元素
        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
        ]):
            tag.decompose()

        # 保留标题层级
        for heading in soup.find_all(
            ["h1", "h2", "h3", "h4"]
        ):
            heading.insert_before("\n\n")
            heading.append("\n")

        return soup.get_text(separator="\n", strip=True)

    def _read_pdf(self, path: Path) -> list[tuple[str, int]]:
        """
        读取 PDF。
        每一页独立返回，保留页码，方便后续引用来源。
        """

        reader = PdfReader(str(path))
        pages = []

        for page_number, page in enumerate(
            reader.pages, start=1
        ):
            text = page.extract_text() or ""

            if text.strip():
                pages.append((text, page_number))

        return pages

    def _read_docx(self, path: Path) -> str:

        """读取 Word 文档中的段落和表格。"""

        doc = WordDocument(str(path))
        parts = []

        # 提取段落
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()

            if text:
                parts.append(text)

        # 提取表格
        for table in doc.tables:
            for row in table.rows:
                cells = [
                    cell.text.strip()
                    for cell in row.cells
                ]
                parts.append(" | ".join(cells))

        return "\n".join(parts)

    def _extract_title(self, content: str, path: Path) -> str:
        """优先从 Markdown 一级标题提取文档标题。"""

        for line in content.splitlines():
            line = line.strip()

            if line.startswith("# "):
                return line[2:].strip()

        return path.stem

    def _build_metadata(
        self,
        path: Path,
        content: str,
        page_number: int | None = None,
    ) -> dict[str, Any]:
        """组合来源登记表和文件本身的信息。"""

        filename = path.name

        # 查找 sources.json 中的登记信息
        registered = self.source_registry.get(
            filename, {}
        )

        # 未登记文件使用文件名作为备用编号
        document_id = registered.get(
            "document_id", path.stem
        )

        # 官方文档使用 URL；模拟文档使用相对路径
        source = registered.get("source_url")

        if not source:
            try:
                source = str(
                    path.relative_to(PROJECT_ROOT)
                )
            except ValueError:
                source = str(path)

        metadata = {
            "document_id": document_id,
            "title": self._extract_title(content, path),
            "file_name": filename,
            "file_path": str(path),
            "source": source,
            "category": registered.get(
                "category", "uncategorized"
            ),
            "language": registered.get(
                "language", "unknown"
            ),
            "document_type": registered.get(
                "document_type", "unknown"
            ),
            "source_type": registered.get(
                "source_type", "unknown"
            ),
            "license": registered.get(
                "license", "unknown"
            ),
            "version": registered.get(
                "version", "unknown"
            ),
        }

        if page_number is not None:
            metadata["page_number"] = page_number

        return metadata

    def _parse_file(
        self, path: Path
    ) -> list[tuple[str, int | None]]:
        """根据扩展名选择对应解析方法。"""

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

    def load_documents(self) -> list[Document]:
        """扫描目录并加载所有支持的文档。"""

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"知识库目录不存在：{self.data_dir}"
            )

        documents = []

        # 递归扫描目录
        for path in sorted(self.data_dir.rglob("*")):

            if not path.is_file():
                continue

            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            try:
                parsed_items = self._parse_file(path)

                for content, page_number in parsed_items:
                    if not content.strip():
                        logger.warning(
                            "文档内容为空，跳过：%s", path
                        )
                        continue

                    metadata = self._build_metadata(
                        path,
                        content,
                        page_number,
                    )

                    document_id = metadata["document_id"]

                    # PDF 每页拥有独立 ID，但保留相同 document_id
                    if page_number is not None:
                        unique_id = (
                            f"{document_id}:page:{page_number}"
                        )
                    else:
                        unique_id = document_id

                    documents.append(
                        Document(
                            id=unique_id,
                            content=content,
                            metadata=metadata,
                        )
                    )

                logger.info("加载成功：%s", path.name)

            except Exception:
                logger.exception(
                    "加载文档失败：%s", path
                )

        logger.info(
            "文档加载完成，共加载 %d 个文档对象",
            len(documents),
        )

        return documents


def load_documents(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    metadata_file: str | Path = DEFAULT_METADATA_FILE,
) -> list[Document]:

    loader = DocumentLoader(
        data_dir=data_dir,
        metadata_file=metadata_file,
    )

    return loader.load_documents()


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    docs = load_documents()


    print(f"\n成功加载 {len(docs)} 个文档对象\n")
    # for doc in docs:
    #     print("=" * 60)
    #     print("文档 ID：", doc.id)
    #     print("标题：", doc.metadata["title"])
    #     print("类型：", doc.metadata["document_type"])
    #     print("语言：", doc.metadata["language"])
    #     print("来源：", doc.metadata["source"])
    #     print("正文长度：", len(doc.content))
    #     print("正文预览：", doc.content[:150])
