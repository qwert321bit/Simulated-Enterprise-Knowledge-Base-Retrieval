"""
文本切分。

1. 按 Markdown 标题识别章节，章节变化时结束当前 Chunk
2. 代码块保持完整，不把代码里的 # 注释当成标题
3. 表格尽量整体保留；超长表格按行拆分，并在每一段重复表头
4. 超长段落按句子切分，相邻 Chunk 保留 chunk_overlap 字符的重叠
5. 每个 Chunk 前统一加一次文档标题和章节上下文
"""

from __future__ import annotations

import re

from backend.core.config import settings
from backend.rag.loader import Document

# 中文标点后直接切；英文句点必须后跟空白才算句末，避免切开 3.11 / .env / example.com
SENTENCE_PATTERN = re.compile(r"(?<=[。！？!?；;])\s*|(?<=[.])\s+")
TABLE_SEPARATOR = re.compile(r"[\s|:\-]+")


class TextSplitter:
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = settings.chunk_size if chunk_size is None else chunk_size
        self.chunk_overlap = (
            settings.chunk_overlap if chunk_overlap is None else chunk_overlap
        )

        if self.chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")

        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须大于等于 0 且小于 chunk_size")

    # ------------------------------------------------------------------
    # 第一步：拆成 段落 / 表格 块
    # ------------------------------------------------------------------
    def _extract_blocks(self, text: str) -> list[dict]:
        blocks: list[dict] = []
        paragraph: list[str] = []
        table: list[str] = []
        headings: list[str] = []
        in_code = False

        def save_paragraph():
            nonlocal paragraph
            content = "\n".join(paragraph).strip()
            if content:
                blocks.append(
                    {"type": "paragraph", "content": content,
                     "heading": " > ".join(headings)}
                )
            paragraph = []

        def save_table():
            nonlocal table
            content = "\n".join(table).strip()
            if content:
                blocks.append(
                    {"type": "table", "content": content,
                     "heading": " > ".join(headings)}
                )
            table = []

        for line in text.splitlines():
            stripped = line.strip()

            # 代码块：整体保留，里面的 # 不是标题
            if stripped.startswith("```"):
                save_table()
                in_code = not in_code
                paragraph.append(line)
                continue

            if in_code:
                paragraph.append(line)
                continue

            heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)

            if heading_match:
                save_paragraph()
                save_table()
                level = len(heading_match.group(1))
                headings = headings[: level - 1]
                headings.append(heading_match.group(2).strip())
                continue

            if stripped.startswith("|") and stripped.endswith("|"):
                save_paragraph()
                table.append(line)
            else:
                save_table()
                if stripped:
                    paragraph.append(line)
                else:
                    save_paragraph()

        save_paragraph()
        save_table()
        return blocks

    # ------------------------------------------------------------------
    # 第二步：超长内容的拆分
    # ------------------------------------------------------------------
    def _overlap_tail(self, units: list[str]) -> list[str]:
        """取末尾若干单元，总长度不超过 chunk_overlap，作为下一个 Chunk 的开头。"""
        tail: list[str] = []
        total = 0

        for unit in reversed(units):
            total += len(unit) + (1 if tail else 0)
            if total > self.chunk_overlap:
                break
            tail.insert(0, unit)

        return tail

    @staticmethod
    def _is_code_block(text: str) -> bool:
        lines = text.splitlines()
        return (
            len(lines) >= 2
            and lines[0].strip().startswith("```")
            and lines[-1].strip() == "```"
        )

    def _split_code(self, text: str) -> list[str]:
        """代码块按行拆分，每一段都补上围栏，保证渲染和阅读时仍是完整代码块。"""
        lines = text.splitlines()
        opening, closing = lines[0].strip(), "```"
        budget = max(self.chunk_size - len(opening) - len(closing) - 2, 1)

        parts: list[list[str]] = []
        current: list[str] = []

        for line in lines[1:-1]:
            if current and len("\n".join(current + [line])) > budget:
                parts.append(current)
                current = []
            current.append(line)

        if current:
            parts.append(current)

        return [f"{opening}\n" + "\n".join(part) + f"\n{closing}" for part in parts]

    def _split_long_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        if self._is_code_block(text):
            return self._split_code(text)

        step = self.chunk_size - self.chunk_overlap
        units: list[str] = []

        for sentence in SENTENCE_PATTERN.split(text):
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) <= self.chunk_size:
                units.append(sentence)
                continue

            # 超长句子：带重叠的滑动窗口
            for start in range(0, len(sentence), step):
                units.append(sentence[start:start + self.chunk_size])
                if start + self.chunk_size >= len(sentence):
                    break

        chunks: list[str] = []
        current: list[str] = []

        for unit in units:
            if current and len("\n".join(current + [unit])) > self.chunk_size:
                chunks.append("\n".join(current))
                current = self._overlap_tail(current)

                if current and len("\n".join(current + [unit])) > self.chunk_size:
                    current = []

            current.append(unit)

        if current:
            chunks.append("\n".join(current))

        return chunks

    def _split_table(self, content: str) -> list[str]:
        rows = content.splitlines()

        header: list[str] = []
        if len(rows) > 2 and TABLE_SEPARATOR.fullmatch(rows[1].strip()):
            header = rows[:2]

        parts: list[str] = []
        current: list[str] = []

        for row in rows[len(header):]:
            if current and len("\n".join(header + current + [row])) > self.chunk_size:
                parts.append("\n".join(header + current))
                current = []
            current.append(row)

        if current:
            parts.append("\n".join(header + current))

        return parts

    # ------------------------------------------------------------------
    # 第三步：聚合成 Chunk
    # ------------------------------------------------------------------
    def _build_chunks(self, blocks: list[dict]) -> list[dict]:
        chunks: list[dict] = []
        buffer = ""
        buffer_heading = ""

        def flush():
            nonlocal buffer
            if buffer.strip():
                chunks.append({"content": buffer.strip(), "heading": buffer_heading})
            buffer = ""

        for block in blocks:
            content = block["content"]
            heading = block["heading"]

            # 章节变化时，先结束当前 Chunk
            if buffer and heading != buffer_heading:
                flush()

            buffer_heading = heading

            if len(content) > self.chunk_size:
                flush()
                pieces = (
                    self._split_table(content)
                    if block["type"] == "table"
                    else self._split_long_text(content)
                )
                chunks.extend({"content": p, "heading": heading} for p in pieces)
                continue

            candidate = f"{buffer}\n\n{content}" if buffer else content

            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                flush()
                buffer_heading = heading
                buffer = content

        flush()
        return chunks

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def split_document(self, document: Document) -> list[Document]:
        blocks = self._extract_blocks(document.content)
        raw_chunks = self._build_chunks(blocks)

        title = document.metadata.get(
            "title", document.metadata.get("file_name", "未知文档")
        )

        result = []

        for index, raw in enumerate(raw_chunks):
            parts = [p.strip() for p in raw["heading"].split(">") if p.strip()]

            # 章节路径第一级与文档标题相同时去掉，避免重复
            if parts and parts[0] == title:
                parts.pop(0)

            heading = " > ".join(parts)

            context = [f"文档标题：{title}"]
            if heading:
                context.append(f"所属章节：{heading}")

            final_content = "\n".join(context) + "\n\n" + raw["content"]

            metadata = document.metadata.copy()
            metadata.update(
                {
                    "chunk_index": index,
                    "chunk_length": len(final_content),
                    "parent_document_id": document.id,
                    "heading": heading,
                }
            )

            result.append(
                Document(
                    id=f"{document.id}:chunk:{index}",
                    content=final_content,
                    metadata=metadata,
                )
            )

        return result

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        for document in documents:
            chunks.extend(self.split_document(document))
        return chunks
