"""应用入口，只负责组装：uvicorn backend.main:app --reload（在项目根目录运行）"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from backend.api import chat, documents
from backend.core.config import settings
from backend.rag.embedding import EmbeddingModel
from backend.rag.generator import RAGGenerator
from backend.rag.reranker import Reranker
from backend.rag.vector_store import QdrantVectorStore
from backend.services.chat_service import ChatService
from backend.services.document_service import DocumentService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化 RAG 系统，关闭时释放 Qdrant 文件锁。"""
    logger.info("正在启动企业知识库问答系统...")
    settings.ensure_dirs()

    vector_store = None

    try:
        embedding_model = EmbeddingModel()
        vector_store = QdrantVectorStore(dimension=embedding_model.dimension)
        reranker = Reranker()

        document_service = DocumentService(embedding_model, vector_store, reranker)
        document_service.load_knowledge_base()

        app.state.document_service = document_service
        app.state.chat_service = ChatService(document_service, RAGGenerator())

        logger.info("企业知识库问答系统启动成功！")
        yield

    finally:
        if vector_store is not None:
            vector_store.close()
        logger.info("企业知识库问答系统已关闭")


app = FastAPI(
    title="Enterprise RAG Assistant",
    description="基于混合检索和 Reranker 的企业知识库问答系统",
    version="1.1.0",
    lifespan=lifespan,
)

app.include_router(chat.router)
app.include_router(documents.router)


@app.get("/health")
async def health_check(request: Request):
    return {
        "status": "ok",
        "service": "Enterprise RAG Assistant",
        "rag_initialized": hasattr(request.app.state, "chat_service"),
    }
