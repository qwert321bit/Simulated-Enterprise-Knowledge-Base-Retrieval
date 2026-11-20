from pydantic import BaseModel


class UploadResponse(BaseModel):
    """文档上传响应"""

    success: bool
    upload_id: str
    file_name: str
    document_count: int
    chunk_count: int
    vector_count: int
    message: str


class DocumentInfo(BaseModel):
    """知识库中的一份文档"""

    document_id: str
    file_name: str
    title: str
    chunk_count: int
    deletable: bool  # 只有通过上传接口加入的文档可以删除
    upload_id: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


class DeleteResponse(BaseModel):
    success: bool
    upload_id: str
    file_name: str
    deleted_chunks: int
    vector_count: int
    message: str
