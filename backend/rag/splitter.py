
"""
RAG 文本切分模块

功能：
1. 按 Markdown 标题识别章节
2. 优先按段落和句子切分
3. 尽量保持 Markdown 表格完整
4. 每个 Chunk 只添加一次标题和章节上下文
5. 保留原始文档元数据
"""

import re
from typing import List

from loader import Document


class TextSplitter:
    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        min_chunk_size: int = 50,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须大于等于 0 且小于 chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def _extract_blocks(self, text: str):
        """
        将 Markdown 文本拆成内容块。
        标题、段落、表格分别处理。
        """

        lines = text.splitlines()

        blocks = []
        current_paragraph = []
        current_table = []
        current_heading = []

        def save_paragraph():
            nonlocal current_paragraph

            content = "\n".join(current_paragraph).strip()

            if content:
                blocks.append({
                    "type": "paragraph",
                    "content": content,
                    "heading": " > ".join(current_heading),
                })

            current_paragraph = []

        def save_table():
            nonlocal current_table

            content = "\n".join(current_table).strip()

            if content:
                blocks.append({
                    "type": "table",
                    "content": content,
                    "heading": " > ".join(current_heading),
                })

            current_table = []

        for line in lines:
            stripped = line.strip()

            # Markdown 标题
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)

            if heading_match:
                save_paragraph()
                save_table()

                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()

                current_heading = current_heading[:level - 1]
                current_heading.append(heading_text)

                continue

            # 判断是否为 Markdown 表格行
            is_table = (
                stripped.startswith("|")
                and stripped.endswith("|")
            )

            if is_table:
                save_paragraph()
                current_table.append(line)

            else:
                save_table()

                if stripped:
                    current_paragraph.append(line)
                else:
                    save_paragraph()

        save_paragraph()
        save_table()

        return blocks

    def _split_long_text(self, text: str) -> List[str]:
        """
        长文本优先按句子切分，超长句子再按字符切分。
        """

        if len(text) <= self.chunk_size:
            return [text]

        sentences = re.split(
            r"(?<=[。！？.!?；;])\s*",
            text
        )

        result = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            if len(sentence) > self.chunk_size:
                if current:
                    result.append(current)
                    current = ""

                # 对超长句子进行硬切
                for i in range(0, len(sentence), self.chunk_size):
                    result.append(
                        sentence[i:i + self.chunk_size]
                    )

                continue

            candidate = (
                current + "\n" + sentence
                if current else sentence
            )

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)

                current = sentence

        if current:
            result.append(current)

        return result

    def _build_chunks(self, blocks):
        """
        聚合内容块，生成最终 Chunk。
        """

        chunks = []
        current_content = ""
        current_heading = ""

        def flush():
            nonlocal current_content

            if current_content.strip():
                chunks.append({
                    "content": current_content.strip(),
                    "heading": current_heading,
                })

            current_content = ""

        for block in blocks:
            content = block["content"]
            heading = block["heading"]

            # 章节变化时，优先结束当前 Chunk
            if current_content and heading != current_heading:
                flush()

            current_heading = heading

            # 表格尽量作为整体保留
            if block["type"] == "table":
                if len(content) > self.chunk_size:
                    # 超长表格只能拆分，尽量保持行完整
                    rows = content.splitlines()
                    table_parts = []
                    part = ""

                    for row in rows:
                        candidate = (
                            part + "\n" + row
                            if part else row
                        )

                        if len(candidate) <= self.chunk_size:
                            part = candidate
                        else:
                            if part:
                                table_parts.append(part)
                            part = row

                    if part:
                        table_parts.append(part)

                    for table_part in table_parts:
                        if current_content:
                            flush()

                        current_heading = heading
                        current_content = table_part
                        flush()

                    continue

            # 普通段落或表格超出 Chunk 大小
            if len(content) > self.chunk_size:
                if current_content:
                    flush()

                parts = self._split_long_text(content)

                for part in parts:
                    current_heading = heading
                    current_content = part
                    flush()

                continue

            candidate = (
                current_content + "\n\n" + content
                if current_content else content
            )

            if len(candidate) <= self.chunk_size:
                current_content = candidate
            else:
                flush()
                current_heading = heading
                current_content = content

        flush()

        return chunks

    def split_document(self, document: Document) -> List[Document]:
        """
        将单篇 Document 切分为多个 Chunk。
        """

        blocks = self._extract_blocks(document.content)
        raw_chunks = self._build_chunks(blocks)

        result = []

        title = document.metadata.get(
            "title",
            document.metadata.get("file_name", "未知文档")
        )

        for index, raw_chunk in enumerate(raw_chunks):
            content = raw_chunk["content"]
            heading = raw_chunk["heading"].strip()

            # 如果章节路径的第一级与文档标题相同，就去掉这一级
            heading_parts = [
                part.strip()
                for part in heading.split(">")
                if part.strip()
            ]

            if heading_parts and heading_parts[0] == title:
                heading_parts.pop(0)

            heading = " > ".join(heading_parts)

            # 统一添加一次上下文
            context_parts = [f"文档标题：{title}"]

            if heading:
                context_parts.append(f"所属章节：{heading}")

            final_content = (
                "\n".join(context_parts)
                + "\n\n"
                + content
            )

            metadata = document.metadata.copy()

            metadata.update({
                "chunk_index": index,
                "chunk_length": len(final_content),
                "parent_document_id": document.id,
                "heading": heading,
            })

            chunk = Document(
                id=f"{document.id}:chunk:{index}",
                content=final_content,
                metadata=metadata,
            )

            result.append(chunk)

        return result

    def split_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """
        批量切分文档。
        """

        all_chunks = []

        for document in documents:
            chunks = self.split_document(document)
            all_chunks.extend(chunks)

        return all_chunks


if __name__ == "__main__":

    from loader import load_documents

    documents = load_documents()

    print(f"原始文档数量：{len(documents)}")

    splitter = TextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        min_chunk_size=50,
    )

    chunks = splitter.split_documents(documents)

    print(f"切分后 Chunk 数量：{len(chunks)}")

    if chunks:
        print("\n第一个 Chunk：")
        print("ID：", chunks[0].id)
        print("标题：", chunks[0].metadata.get("title"))
        print("章节：", chunks[0].metadata.get("heading"))
        print("长度：", chunks[0].metadata.get("chunk_length"))
        print("内容：\n", chunks[0].content)