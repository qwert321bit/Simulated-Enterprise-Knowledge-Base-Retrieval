# E004 星云科技 Docker 部署手册

> **模拟企业资料声明**：本文档中的企业、服务名、网络、端口和部署流程均为项目原创模拟内容。示例配置不能直接用于真实生产环境。

| 属性 | 内容 |
|---|---|
| 文档编号 | E004 |
| 版本 | 1.0 |
| 发布日期 | 2026-11-15 |
| 适用部门 | 平台工程、后端研发、测试 |
| 部署对象 | NebulaTech Knowledge Service |

## 1. 部署架构

演示环境由以下服务组成：

| 服务 | 用途 | 容器端口示例 |
|---|---|---|
| `api` | FastAPI 应用 | 8000 |
| `postgres` | 关系型数据存储 | 5432 |
| `redis` | 缓存和短期状态 | 6379 |
| `qdrant` | 向量检索 | 6333 |

容器间通信使用 Docker Compose 网络中的服务名。应用容器连接数据库时应使用 `postgres` 作为主机名，而不是容器内的 `localhost`。

## 2. 部署前检查

- 已安装受支持的 Docker Engine / Docker Desktop 和 Compose。
- 已准备 `.env`，其中包含数据库和模型服务配置。
- `.env` 不得提交到 Git。
- 镜像版本应固定，不使用不受控的 `latest`。
- 已确认宿主机端口未被其他程序占用。
- 数据卷有明确的持久化和备份策略。

本地开发环境准备参照 E001。

## 3. Compose 配置示例

以下为结构示例。镜像版本、密码和密钥必须按实际环境替换；占位符不是可用凭据。

```yaml
services:
  api:
    build: .
    env_file:
      - .env
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
      - qdrant
    restart: unless-stopped

  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: <DB_NAME>
      POSTGRES_USER: <DB_USER>
      POSTGRES_PASSWORD: <DB_PASSWORD>
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7
    volumes:
      - redis_data:/data

  qdrant:
    image: qdrant/qdrant:<PINNED_VERSION>
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  postgres_data:
  redis_data:
  qdrant_data:
```

注意：上述配置用于说明服务关系。真实部署还需补充健康检查、网络隔离、资源限制、认证、备份和安全配置。不要把占位符直接当作 Compose 的有效密码值。

## 4. 常用部署命令

```bash
docker compose config
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f api
docker compose down
```

`docker compose down` 默认不会删除命名卷；执行 `docker compose down -v` 会删除 Compose 管理的卷，可能造成数据丢失。未经备份和审批禁止在有价值数据的环境中执行 `-v`。

## 5. 健康检查与验收

1. 使用 `docker compose ps` 检查容器状态。
2. 查看 API 日志，确认应用成功启动。
3. 请求 `http://localhost:8000/health`（前提是应用实现了该演示接口）。
4. 检查 PostgreSQL、Redis 和 Qdrant 的连接。
5. 上传一份测试文档并执行一次检索。
6. 检查重启后持久化数据是否仍然存在。

## 6. 常见故障排查

| 故障 | 可能原因 | 排查方法 |
|---|---|---|
| API 容器反复重启 | 环境变量缺失、启动命令错误、依赖不可用 | `docker compose logs api` |
| API 连接数据库失败 | 使用了错误主机名、凭据错误、数据库未就绪 | 检查 `DATABASE_URL`，容器网络中使用 `postgres` |
| 端口绑定失败 | 宿主机端口被占用 | 更换宿主机映射端口或停止冲突进程 |
| Qdrant 数据丢失 | 未使用持久化卷或卷被删除 | 检查 `qdrant_data` 挂载和备份 |
| 镜像构建失败 | 网络、依赖或 Dockerfile 错误 | 查看 build 输出，单独验证依赖安装 |
| 容器能启动但接口不可用 | 应用监听地址错误 | 容器内服务通常需监听 `0.0.0.0` |

## 7. 更新与回滚

1. 确认目标镜像版本和变更说明。
2. 备份数据库和需要保留的向量数据。
3. 在测试环境验证新版本。
4. 执行部署并观察日志、健康状态和关键接口。
5. 失败时回滚到上一个已验证镜像版本。
6. 若涉及数据库 Schema 变更，必须先确认迁移是否可逆。

## 8. 日志与监控

日志应包含时间、级别、服务名和 request_id。禁止记录 API Key、Authorization Header、用户密码或完整敏感文档内容。故障排查流程参照 E006，敏感信息处理参照 E007。

## 9. 关联文档

- E001：开发环境和环境变量。
- E003：部署变更的代码审查要求。
- E005：数据库备份与恢复。
- E006：容器故障处理。
- E007：部署凭据和日志脱敏。
