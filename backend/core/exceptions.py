"""项目自定义异常，API 层据此映射 HTTP 状态码。"""


class DocumentError(Exception):
    """文档处理相关错误的基类。"""


class DuplicateDocumentError(DocumentError):
    """文件内容已经在知识库中（409）。"""


class UnprocessableDocumentError(DocumentError):
    """文件无法解析或没有有效内容，例如扫描版 PDF（422）。"""


class DocumentNotFoundError(DocumentError):
    """要删除的文档不存在（404）。"""


class LLMError(Exception):
    """大模型调用失败或返回空答案（502）。"""
