# E002 星云科技 API 接口开发规范

> **模拟企业资料声明**：本文档为项目原创模拟资料。星云科技、接口路径、字段、业务规则和示例数据均为虚构内容。

| 属性 | 内容 |
|---|---|
| 文档编号 | E002 |
| 版本 | 1.0 |
| 发布日期 | 2026-11-15 |
| 适用部门 | 后端研发、前端研发、测试 |
| 关联系统 | NebulaTech Knowledge Service |

## 1. 目的

本规范统一 HTTP API 的设计、实现、错误处理和版本管理方式。后端默认使用 Python 3.11 与 FastAPI。接口必须可预测、可测试，并提供足够的日志与文档。

## 2. 通用约定

- API 前缀：`/api/v1`。
- 数据交换格式：UTF-8 编码 JSON。
- 时间格式：RFC 3339 / ISO 8601，统一使用带时区的时间戳。
- 字段命名：JSON 使用 `snake_case`。
- 成功响应应使用明确的 HTTP 状态码。
- 不得在响应体、URL 或日志中输出 API Key、密码、访问令牌等秘密。

## 3. HTTP 方法选择

| 方法 | 典型用途 | 幂等性约定 |
|---|---|---|
| GET | 查询资源 | 应为幂等 |
| POST | 创建资源或执行非幂等操作 | 默认不保证幂等 |
| PUT | 完整替换资源 | 应为幂等 |
| PATCH | 局部更新资源 | 需明确更新语义 |
| DELETE | 删除资源 | 应设计为重复调用结果可预测 |

创建知识库文档使用 `POST /api/v1/documents`；查询文档使用 `GET /api/v1/documents/{document_id}`；删除文档使用 `DELETE /api/v1/documents/{document_id}`。

## 4. 请求与响应示例

创建文档请求：

```http
POST /api/v1/documents
Content-Type: application/json
Authorization: Bearer <ACCESS_TOKEN>
```

```json
{
  "file_name": "developer-guide.md",
  "category": "software-engineering",
  "language": "zh"
}
```

成功响应：

```json
{
  "document_id": "doc_demo_001",
  "status": "accepted",
  "message": "Document queued for ingestion"
}
```

`doc_demo_001` 是虚构示例 ID，不对应真实资源。

## 5. 状态码约定

| 状态码 | 含义 | 使用场景 |
|---|---|---|
| 200 | OK | 查询或更新成功 |
| 201 | Created | 资源创建成功 |
| 202 | Accepted | 异步任务已接收 |
| 400 | Bad Request | 请求格式或参数不合法 |
| 401 | Unauthorized | 未认证或令牌无效 |
| 403 | Forbidden | 已认证但无权限 |
| 404 | Not Found | 资源不存在 |
| 409 | Conflict | 资源状态冲突或重复操作 |
| 422 | Unprocessable Entity | 请求字段校验失败 |
| 429 | Too Many Requests | 请求频率超限 |
| 500 | Internal Server Error | 未预期的服务端错误 |
| 503 | Service Unavailable | 依赖服务暂时不可用 |

不得将所有异常统一返回 200。客户端应根据状态码采取相应处理。

## 6. 统一错误结构

```json
{
  "error": {
    "code": "DOCUMENT_NOT_FOUND",
    "message": "The requested document does not exist",
    "request_id": "req_demo_123"
  }
}
```

错误消息应帮助调用方定位问题，但不得泄露 SQL、堆栈、服务器内部路径或秘密信息。`request_id` 用于关联日志。

## 7. FastAPI 实现要求

- 使用 Pydantic 模型校验输入和输出。
- 路由层负责 HTTP 参数和状态码；业务逻辑放在 service 层。
- 通过依赖注入提供数据库会话、认证用户和配置。
- 为外部依赖设置超时。
- 对分页接口设置合理的默认值和最大值。
- 对上传接口限制文件大小、扩展名和 MIME 类型；不能仅凭文件扩展名判断文件安全。

## 8. 分页与筛选

列表接口采用 `page` 和 `page_size` 参数：

```http
GET /api/v1/documents?page=1&page_size=20&category=python
```

约定：`page` 从 1 开始；`page_size` 默认 20，最大 100。响应应包含 `items`、`page`、`page_size` 和 `total`。

## 9. 认证、授权与限流

- 需要用户身份的接口必须执行认证。
- 认证通过不代表拥有所有资源的访问权限。
- 对上传、检索和模型生成接口设置用户级或租户级限流。
- 返回 429 时可提供 `Retry-After`。
- 访问控制规则参照 E007《数据安全管理规范》。

## 10. 接口变更流程

1. 提交接口变更说明和 OpenAPI Schema。
2. 评估对调用方的兼容性影响。
3. 更新自动化测试和接口文档。
4. 按 E003 流程完成代码审查。
5. 对不兼容变更升级 API 主版本或提供迁移窗口。

## 11. 测试要求

至少覆盖正常请求、缺少字段、非法字段、未认证、无权限、资源不存在、重复请求、限流和依赖服务不可用等情况。

## 12. 关联文档

- E001：本地开发环境与配置。
- E003：代码审查、测试和提交要求。
- E006：接口超时和服务异常排查。
- E007：认证、授权和敏感信息保护。
- E008：大模型 API 调用约定。
