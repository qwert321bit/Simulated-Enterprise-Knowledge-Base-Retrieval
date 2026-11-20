"""
Streamlit 前端：问答、来源展示、文档上传与删除。

    streamlit run frontend/app.py

后端地址通过环境变量 BACKEND_URL 配置，默认 http://127.0.0.1:8000。
"""

import json
import os

import requests
import streamlit as st
from collections.abc import Iterator

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

# 后端在 CPU 上做重排序和调用大模型，耗时较长，超时要给足
CHAT_TIMEOUT = 180
UPLOAD_TIMEOUT = 600
SHORT_TIMEOUT = 10

st.set_page_config(page_title="企业知识库助手", page_icon="📚", layout="wide")


# ----------------------------------------------------------------------
# 后端调用
# ----------------------------------------------------------------------
class BackendError(Exception):
    """后端返回了错误，message 可以直接展示给用户。"""


def error_text(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        return f"HTTP {response.status_code}"

    if isinstance(detail, list):  # FastAPI 的参数校验错误
        return "；".join(str(item.get("msg", item)) for item in detail)

    return str(detail)


def call(method: str, path: str, timeout: int = SHORT_TIMEOUT, **kwargs) -> dict:
    try:
        response = requests.request(method, f"{BACKEND_URL}{path}", timeout=timeout, **kwargs)
    except requests.ConnectionError:
        raise BackendError(f"无法连接后端：{BACKEND_URL}，请确认服务已启动。")
    except requests.Timeout:
        raise BackendError("后端响应超时，请稍后重试。")

    if not response.ok:
        raise BackendError(error_text(response))

    return response.json()


def backend_online() -> bool:
    try:
        return call("GET", "/health", timeout=3).get("rag_initialized", False)
    except BackendError:
        return False


def stream_answer(query: str, top_k: int, holder: dict, status) -> "Iterator[str]":
    """
    调用 /chat/stream，逐段产出答案文本。
    来源信息先于答案到达，写进 holder 供答案结束后展示；status 是等待提示的占位符。
    """
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat/stream",
            json={"query": query, "top_k": top_k},
            stream=True,
            # (连接超时, 两次数据之间的最长等待)
            timeout=(SHORT_TIMEOUT, CHAT_TIMEOUT),
        )
    except requests.ConnectionError:
        raise BackendError(f"无法连接后端：{BACKEND_URL}，请确认服务已启动。")
    except requests.Timeout:
        raise BackendError("后端响应超时，请稍后重试。")

    if not response.ok:
        raise BackendError(error_text(response))

    event = None

    try:
        # chunk_size=None：数据到达就立即处理，默认的 512 字节会把短 token 攒住再吐出
        for raw in response.iter_lines(chunk_size=None):
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw

            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())

                if event == "sources":
                    holder.update(data)
                    status.empty()
                elif event == "token":
                    yield data["text"]
                elif event == "error":
                    raise BackendError(data["message"])
                elif event == "done":
                    return
    except requests.RequestException:
        raise BackendError("与后端的连接中断，请稍后重试。")
    finally:
        response.close()


# ----------------------------------------------------------------------
# 页面状态
# ----------------------------------------------------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploader_key", 0)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    with st.expander(f"引用来源（{len(sources)}）"):
        for s in sources:
            line = f"**[{s['index']}] {s['title']}**  \n`{s['source']}`"

            if s.get("heading"):
                line += f"  \n章节：{s['heading']}"
            if s.get("page_number") is not None:
                line += f"  \n页码：{s['page_number']}"
            if s.get("rerank_score") is not None:
                line += f"  \n相关度：{s['rerank_score']:.3f}"

            st.markdown(line)


# ----------------------------------------------------------------------
# 侧边栏：状态、参数、文档管理
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("📚 知识库")

    online = backend_online()
    if online:
        st.success("后端已就绪", icon="✅")
    else:
        st.error("后端未就绪（启动中或未运行）", icon="⚠️")

    top_k = st.slider("引用片段数量（top_k）", min_value=1, max_value=10, value=3)

    if st.button("清空对话"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("上传文档")

    uploaded = st.file_uploader(
        "支持 PDF / DOCX / TXT / MD，单个文件不超过 10 MB",
        type=["pdf", "docx", "txt", "md"],
        key=f"uploader_{st.session_state.uploader_key}",
        disabled=not online,
    )

    if uploaded is not None and st.button("加入知识库", type="primary"):
        with st.spinner("正在解析并向量化，请稍候..."):
            try:
                result = call(
                    "POST",
                    "/documents/upload",
                    timeout=UPLOAD_TIMEOUT,
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                )
                st.success(f"已加入：{result['file_name']}（{result['chunk_count']} 个片段）")
                st.session_state.uploader_key += 1
                st.rerun()
            except BackendError as e:
                st.error(str(e))

    st.divider()
    st.subheader("文档列表")

    if online:
        try:
            documents = call("GET", "/documents")["documents"]
        except BackendError as e:
            documents = []
            st.warning(str(e))

        if not documents:
            st.caption("知识库里还没有文档。")

        for doc in documents:
            label = f"{doc['file_name'] or doc['title']}　·　{doc['chunk_count']} 片段"
            with st.expander(label):
                st.caption(doc["title"])

                if doc["deletable"]:
                    if st.button("删除", key=f"delete_{doc['upload_id']}"):
                        try:
                            call("DELETE", f"/documents/{doc['upload_id']}")
                            st.rerun()
                        except BackendError as e:
                            st.error(str(e))
                else:
                    st.caption("系统内置文档，不可在页面删除。")


# ----------------------------------------------------------------------
# 主区域：对话
# ----------------------------------------------------------------------
st.title("企业知识库问答助手")
st.caption("基于混合检索（向量 + BM25）、Reranker 重排序和大模型生成；回答只依据知识库资料。")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

query = st.chat_input("请输入你的问题", disabled=not online)

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        holder: dict = {}
        status = st.empty()
        status.caption("正在检索知识库...")

        try:
            # write_stream 边接收边显示，结束后返回完整答案
            answer = st.write_stream(stream_answer(query, top_k, holder, status))
            sources = holder.get("sources", [])
        except BackendError as e:
            status.empty()
            answer, sources = f"⚠️ {e}", []
            st.markdown(answer)

        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
