import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.api.sse import stream_events
from backend.core.config import settings
from backend.core.exceptions import LLMError
from backend.schemas.chat import ChatRequest, ChatResponse
from backend.services.chat_service import ChatService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])


def get_chat_service(request: Request) -> ChatService:
    service = getattr(request.app.state, "chat_service", None)

    if service is None:
        raise HTTPException(status_code=503, detail="RAG 系统尚未初始化")

    return service


def to_http_error(exc: Exception) -> HTTPException:
    """把服务层异常映射成 HTTP 错误；上游细节只写日志，不返回给客户端。"""
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))

    if isinstance(exc, LLMError):
        logger.error("大模型调用失败：%s", exc)
        return HTTPException(status_code=502, detail="大模型服务暂时不可用，请稍后重试")

    logger.error("问答接口发生错误", exc_info=exc)
    return HTTPException(status_code=500, detail="知识库问答服务发生内部错误")


# 注意是普通 def：FastAPI 会放进线程池执行，不会阻塞事件循环
@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, service: ChatService = Depends(get_chat_service)):
    """接收问题并调用 RAG 系统，一次性返回完整答案。"""
    try:
        result = service.ask(
            query=body.query,
            top_k=body.top_k,
            candidate_k=settings.candidate_k,
        )
    except Exception as e:
        raise to_http_error(e)

    return ChatResponse(
        query=result["query"],
        answer=result["answer"],
        sources=result["sources"],
        retrieved_count=len(result["retrieved_chunks"]),
    )


@router.post("/chat/stream")
def chat_stream(body: ChatRequest, service: ChatService = Depends(get_chat_service)):
    """
    流式问答（Server-Sent Events）。事件：
    - sources: {"query", "sources", "retrieved_count"}，先于答案发送
    - token:   {"text"}，答案的一段文本
    - done:    {}，正常结束
    - error:   {"message"}，流开始后出错
    """
    try:
        result = service.ask_stream(
            query=body.query,
            top_k=body.top_k,
            candidate_k=settings.candidate_k,
        )

        # 先取出第一段文本：检索失败、鉴权失败、连接超时这类错误
        # 在响应开始前就能暴露，仍然可以返回正确的 HTTP 状态码
        tokens = iter(result.tokens)
        first_token = next(tokens, None)

    except Exception as e:
        raise to_http_error(e)

    start = {
        "query": result.query,
        "sources": result.sources,
        "retrieved_count": result.retrieved_count,
    }

    return StreamingResponse(
        stream_events(start, first_token, tokens),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
