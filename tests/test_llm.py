from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAIError

from backend.core import llm as llm_module
from backend.core.exceptions import LLMError


class FakeStream:
    """模拟 openai 的流式响应：可迭代，并且有 close()。"""

    def __init__(self, events, error=None):
        self.events = events
        self.error = error
        self.closed = False

    def __iter__(self):
        yield from self.events
        if self.error:
            raise self.error

    def close(self):
        self.closed = True


def event(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


FINAL_EVENT = SimpleNamespace(choices=[])  # 有些服务商的最后一个事件 choices 为空


def install_fake_openai(monkeypatch, content="  你好  ", error=None, stream=None):
    """把 llm 模块里的 OpenAI 换成假客户端。stream 是预置的 FakeStream。"""
    created = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            created["init"] = kwargs
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            created["request"] = kwargs
            if error:
                raise error
            if kwargs.get("stream"):
                return stream
            message = SimpleNamespace(content=content)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(llm_module, "OpenAI", FakeOpenAI)
    return created


def test_missing_api_key_raises(monkeypatch, override_settings):
    override_settings(deepseek_api_key=None)
    install_fake_openai(monkeypatch)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        llm_module.LLMClient()


def test_chat_returns_stripped_answer(monkeypatch):
    created = install_fake_openai(monkeypatch, content="  你好  ")
    client = llm_module.LLMClient(api_key="k", base_url="http://x", model="m")

    assert client.chat("sys", "user") == "你好"
    assert created["request"]["model"] == "m"
    assert created["request"]["messages"][0] == {"role": "system", "content": "sys"}


def test_client_gets_timeout_and_retries(monkeypatch, override_settings):
    override_settings(llm_timeout=12.5, llm_max_retries=4)
    created = install_fake_openai(monkeypatch)
    llm_module.LLMClient(api_key="k")

    assert created["init"]["timeout"] == 12.5
    assert created["init"]["max_retries"] == 4


def test_empty_answer_raises(monkeypatch):
    install_fake_openai(monkeypatch, content="   ")
    client = llm_module.LLMClient(api_key="k")

    with pytest.raises(LLMError, match="空答案"):
        client.chat("sys", "user")


def test_provider_error_is_wrapped(monkeypatch):
    install_fake_openai(monkeypatch, error=OpenAIError("boom"))
    client = llm_module.LLMClient(api_key="k")

    with pytest.raises(LLMError, match="boom"):
        client.chat("sys", "user")


# ----------------------------- 流式 -----------------------------
def test_stream_yields_deltas_and_skips_empty_events(monkeypatch):
    stream = FakeStream([event("你"), event(None), event(""), event("好"), FINAL_EVENT])
    created = install_fake_openai(monkeypatch, stream=stream)
    client = llm_module.LLMClient(api_key="k")

    assert list(client.stream_chat("sys", "user")) == ["你", "好"]
    assert created["request"]["stream"] is True
    assert stream.closed


def test_stream_without_any_text_raises(monkeypatch):
    install_fake_openai(monkeypatch, stream=FakeStream([event(None), FINAL_EVENT]))
    client = llm_module.LLMClient(api_key="k")

    with pytest.raises(LLMError, match="空答案"):
        list(client.stream_chat("sys", "user"))


def test_stream_connection_error_surfaces_on_first_iteration(monkeypatch):
    install_fake_openai(monkeypatch, error=OpenAIError("auth failed"))
    client = llm_module.LLMClient(api_key="k")

    generator = client.stream_chat("sys", "user")  # 创建生成器不会触发请求
    with pytest.raises(LLMError, match="auth failed"):
        next(generator)


def test_stream_midway_network_error_is_wrapped(monkeypatch):
    stream = FakeStream([event("前半")], error=httpx.ReadError("connection reset"))
    install_fake_openai(monkeypatch, stream=stream)
    client = llm_module.LLMClient(api_key="k")

    generator = client.stream_chat("sys", "user")
    assert next(generator) == "前半"

    with pytest.raises(LLMError, match="connection reset"):
        next(generator)

    assert stream.closed


def test_stream_closes_upstream_when_consumer_stops_early(monkeypatch):
    stream = FakeStream([event("a"), event("b"), event("c")])
    install_fake_openai(monkeypatch, stream=stream)
    client = llm_module.LLMClient(api_key="k")

    generator = client.stream_chat("sys", "user")
    next(generator)
    generator.close()  # 模拟客户端断开

    assert stream.closed
