# E001 星云科技研发环境配置手册

> **模拟企业资料声明**：本文档由 Enterprise RAG Assistant 项目原创编写，虚构企业“星云科技（NebulaTech）”及其流程、系统、人员、地址和配置均为演示用途，不代表任何真实企业制度。

| 属性 | 内容 |
|---|---|
| 文档编号 | E001 |
| 版本 | 1.0 |
| 发布日期 | 2026-11-15 |
| 适用部门 | 研发部、测试部、平台工程组 |
| 文档负责人 | 模拟平台工程组 |
| 密级 | 项目演示资料 |

## 1. 目的与适用范围

本手册定义星云科技开发人员的本地研发环境基线，确保开发、测试和代码审查使用一致的运行环境。适用于 Python 后端服务、FastAPI API 服务以及调用大模型 API 的内部演示应用。

本手册不规定真实生产环境的账号或密码。所有主机名、IP、令牌和密钥均为占位符。

## 2. 标准技术基线

| 组件 | 标准版本或约定 | 用途 |
|---|---|---|
| Python | 3.11.x | 后端运行时 |
| pip | 与 Python 3.11 配套的近期版本 | Python 包管理 |
| FastAPI | 项目锁定的兼容版本 | HTTP API |
| PostgreSQL | 16.x | 关系型数据 |
| Redis | 7.x | 缓存与短期状态 |
| Qdrant | 以项目部署清单锁定的版本为准 | 向量检索 |
| Docker / Compose | 使用受支持的稳定版本 | 本地容器化 |
| Git | 受支持的稳定版本 | 版本控制 |

依赖版本必须记录在 `requirements.txt` 或锁文件中。禁止在生产部署时依赖未固定版本的 `latest` 标签。版本升级应按 E003 的代码审查流程执行。

## 3. Windows 开发环境初始化

在 PowerShell 中进入项目目录：

```powershell
cd D:\projects\nebulatech-service
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果 PowerShell 阻止当前会话执行虚拟环境激活脚本，可仅对当前进程调整执行策略：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

不得为了激活虚拟环境而无必要地修改整台计算机的全局执行策略。

## 4. Linux / macOS 开发环境初始化

```bash
python3.11 --version
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. 环境变量管理

应用配置通过环境变量注入。开发者应复制 `.env.example` 为本地 `.env`，再填写个人测试配置：

```dotenv
APP_ENV=development
APP_PORT=8000
DATABASE_URL=postgresql://<USER>:<PASSWORD>@<DB_HOST>:5432/<DB_NAME>
REDIS_URL=redis://<REDIS_HOST>:6379/0
QDRANT_URL=http://<QDRANT_HOST>:6333
LLM_API_KEY=<YOUR_API_KEY>
LLM_BASE_URL=<PROVIDER_BASE_URL>
LLM_MODEL=<MODEL_NAME>
```

以上内容仅为格式示例。尖括号中的值必须替换为本地或测试环境配置。不得提交真实 `.env` 文件、API Key、数据库密码或访问令牌。

项目 `.gitignore` 至少应包含：

```gitignore
.env
.env.*
!.env.example
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
data/uploads/
```

如需共享配置，只能提交不含秘密的 `.env.example`。

## 6. 本地服务启动顺序

1. 启动 PostgreSQL、Redis 和 Qdrant 等依赖服务。
2. 检查环境变量是否加载成功。
3. 执行数据库迁移（如项目启用了迁移工具）。
4. 启动 FastAPI 服务。
5. 访问健康检查接口确认服务状态。

示例：

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

健康检查地址示例：`http://127.0.0.1:8000/health`。该路径为本项目约定的演示接口，若具体服务未实现，应先补充接口，不得假定它自动存在。

## 7. 常见问题排查

| 现象 | 可能原因 | 处理方式 |
|---|---|---|
| `python` 不是内部命令 | Python 未加入 PATH 或虚拟环境未激活 | 使用 `py -3.11` 检查并重新激活 `.venv` |
| `ModuleNotFoundError` | 当前解释器与安装依赖的解释器不一致 | 检查 `where python`，在激活的环境中安装依赖 |
| API Key 读取为 `None` | `.env` 路径错误或变量名不一致 | 检查工作目录、`load_dotenv()` 和变量名 |
| 连接数据库失败 | 服务未启动、地址错误或凭据不匹配 | 检查服务状态、连接串和网络 |
| Qdrant 无法连接 | 容器未启动或端口映射不正确 | 检查 Compose 状态和 `QDRANT_URL` |

## 8. 环境验收清单

- [ ] Python 版本为 3.11.x。
- [ ] 虚拟环境可以正常激活。
- [ ] 依赖安装成功，且版本已记录。
- [ ] `.env` 不在 Git 跟踪范围内。
- [ ] PostgreSQL、Redis、Qdrant 连接配置正确。
- [ ] FastAPI 服务能启动并通过健康检查。
- [ ] 开发者能够说明本地配置与测试配置的区别。

## 9. 关联文档

- E002《API 接口开发规范》：定义接口路径、请求与响应格式。
- E003《代码审查与 Git 提交规范》：定义依赖变更和代码合并要求。
- E004《Docker 部署手册》：定义容器化运行方式。
- E007《数据安全管理规范》：定义密钥和敏感数据管理要求。
