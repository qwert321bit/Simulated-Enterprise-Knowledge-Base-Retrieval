"""Server-Sent Events 编码，以及把答案流转换成事件流。"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

from backend.core.exceptions import LLMError

logger = logging.getLogger(__name__)


def encode_event(event: str, data: dict[str, Any]) -> str:
    """一个 SSE 事件。data 是单行 JSON，换行会被转义，不会破坏帧格式。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def stream_events(
    start: dict[str, Any],
    first_token: str | None,
    tokens: Iterator[str],
) -> Iterator[str]:
    """
    事件顺序：sources -> token * N -> done。
    流开始后无法再改 HTTP 状态码，所以中途出错用 error 事件通知前端，
    并且不给上游异常的细节，只写日志。
    """
    try:
        yield encode_event("sources", start)

        if first_token is not None:
            yield encode_event("token", {"text": first_token})

        for token in tokens:
            yield encode_event("token", {"text": token})

        yield encode_event("done", {})

    except LLMError as e:
        logger.error("流式生成中断：%s", e)
        yield encode_event("error", {"message": "大模型服务中断，请稍后重试"})

    except Exception:
        logger.exception("流式生成失败")
        yield encode_event("error", {"message": "生成答案时发生内部错误"})

    finally:
        # 客户端断开时生成器会被关闭，这里把上游的大模型连接一并关掉
        close = getattr(tokens, "close", None)
        if callable(close):
            close()
