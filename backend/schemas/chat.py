from pydantic import BaseModel, Field

from backend.core.config import settings


class ChatRequest(BaseModel):
    """用户提问请求"""

    query: str = Field(
        ...,
        min_length=1,
        description="用户的问题",
        examples=["星云科技的研发环境需要安装什么 Python 版本？"],
    )
    top_k: int = Field(
        default=settings.default_top_k,
        ge=1,
        le=10,
        description="最终交给大模型的知识片段数量",
    )


class Source(BaseModel):
    """答案引用的来源"""

    index: int
    title: str
    source: str
    heading: str | None = None
    page_number: int | None = None
    chunk_id: str | None = None
    rerank_score: float | None = None


class ChatResponse(BaseModel):
    """问答响应"""

    query: str
    answer: str
    sources: list[Source]
    retrieved_count: int
