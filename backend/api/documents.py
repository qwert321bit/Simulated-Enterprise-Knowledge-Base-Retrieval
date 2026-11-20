import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from backend.core.config import settings
from backend.core.exceptions import (
    DocumentError,
    DocumentNotFoundError,
    DuplicateDocumentError,
)
from backend.schemas.document import (
    DeleteResponse,
    DocumentListResponse,
    UploadResponse,
)
from backend.services.document_service import DocumentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


def get_document_service(request: Request) -> DocumentService:
    service = getattr(request.app.state, "document_service", None)

    if service is None:
        raise HTTPException(status_code=503, detail="RAG 系统尚未初始化")

    return service


def _safe_filename(raw_name: str | None) -> str:
    if not raw_name:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    # 兼容 Windows 路径分隔符，防止路径穿越
    name = raw_name.replace("\\", "/").split("/")[-1].strip()

    if not name:
        raise HTTPException(status_code=400, detail="文件名无效")

    if Path(name).suffix.lower() not in settings.upload_extensions:
        raise HTTPException(status_code=400, detail="仅支持 PDF、DOCX、TXT、MD 文件")

    return name


# 普通 def：文件读写和入库都是阻塞操作，交给线程池
@router.post("/upload", response_model=UploadResponse)
def upload_document(
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
):
    """上传文档并加入 RAG 知识库。"""
    safe_name = _safe_filename(file.filename)

    # 每次上传一个独立目录，文件保持原名，目录名即 upload_id
    upload_id = uuid.uuid4().hex
    target_dir = settings.upload_dir / upload_id
    target_path = target_dir / safe_name

    try:
        target_dir.mkdir(parents=True, exist_ok=True)

        total_size = 0
        with target_path.open("wb") as out:
            while chunk := file.file.read(settings.upload_read_chunk):
                total_size += len(chunk)

                if total_size > settings.max_upload_size:
                    raise HTTPException(status_code=413, detail="文件大小不能超过 10 MB")

                out.write(chunk)

        if total_size == 0:
            raise HTTPException(status_code=400, detail="上传文件不能为空")

        result = service.add_document(
            target_path, original_name=safe_name, upload_id=upload_id
        )

        return UploadResponse(success=True, upload_id=upload_id, **result)

    except DuplicateDocumentError as e:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise HTTPException(status_code=409, detail=str(e))

    except DocumentError as e:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise HTTPException(status_code=422, detail=str(e))

    except HTTPException:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise

    except Exception:
        shutil.rmtree(target_dir, ignore_errors=True)
        logger.exception("文档上传处理失败")
        raise HTTPException(status_code=500, detail="文档处理失败，请检查服务端日志")

    finally:
        file.file.close()


@router.get("", response_model=DocumentListResponse)
def list_documents(service: DocumentService = Depends(get_document_service)):
    """列出知识库中的全部文档。"""
    documents = service.list_documents()
    return DocumentListResponse(documents=documents, total=len(documents))


@router.delete("/{upload_id}", response_model=DeleteResponse)
def delete_document(
    upload_id: str,
    service: DocumentService = Depends(get_document_service),
):
    """删除一份上传的文档，并同步清理它的 Chunk 和向量。"""
    try:
        result = service.delete_document(upload_id)
    except DocumentNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        logger.exception("删除文档失败")
        raise HTTPException(status_code=500, detail="删除文档失败，请检查服务端日志")

    return DeleteResponse(success=True, **result)
