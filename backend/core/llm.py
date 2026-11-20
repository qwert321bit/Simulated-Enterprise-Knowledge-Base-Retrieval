"""大模型客户端封装：统一超时、重试和错误类型。"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import httpx
from openai import OpenAI, OpenAIError

from backend.core.config import settings
from backend.core.exceptions import LLMError

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI 兼容接口（默认 DeepSeek）。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or settings.deepseek_api_key
        self.base_url = base_url or settings.deepseek_base_url
        self.model = model or settings.deepseek_model

        if not self.api_key:
            raise ValueError(
                "没有读取到 DEEPSEEK_API_KEY，请检查项目根目录下的 .env 文件"
            )

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )
        logger.info("LLM 客户端初始化成功，模型：%s", self.model)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                stream=False,
            )
        except OpenAIError as e:
            raise LLMError(f"大模型调用失败：{e}") from e

        answer = (response.choices[0].message.content or "").strip()

        if not answer:
            raise LLMError("大模型返回了空答案")

        return answer

    def stream_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        """
        流式调用，逐段产出文本。

        注意这是生成器：连接、鉴权等错误要等到第一次迭代才会抛出。
        调用方提前停止迭代（例如客户端断开）时会关闭上游连接。
        """
        stream = None
        produced = False

        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                stream=True,
            )

            for event in stream:
                # 最后一个事件的 choices 可能为空，增量里的 content 也可能是 None
                if not event.choices:
                    continue

                delta = event.choices[0].delta.content

                if delta:
                    produced = True
                    yield delta

        except (OpenAIError, httpx.HTTPError) as e:
            raise LLMError(f"大模型调用失败：{e}") from e

        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

        if not produced:
            raise LLMError("大模型返回了空答案")
