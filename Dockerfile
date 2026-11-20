# syntax=docker/dockerfile:1
# 同一个 Dockerfile 构建两个镜像：--target backend / --target frontend

FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app


# ---------------------------- 后端 ----------------------------
FROM base AS backend

# 先装 CPU 版 PyTorch，镜像比默认的 CUDA 版小很多；有 GPU 需求时删掉这一行
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY backend ./backend
COPY scripts ./scripts

EXPOSE 8000

# 启动时要加载模型并同步向量库，首次还要下载模型，所以给足 start_period
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

# Qdrant 本地模式是文件锁，只能单进程，不要加 --workers
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]


# ---------------------------- 前端 ----------------------------
FROM base AS frontend

COPY requirements-frontend.txt .
RUN pip install -r requirements-frontend.txt

COPY frontend ./frontend

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
