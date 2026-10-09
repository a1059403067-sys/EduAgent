# EduAgent — AI 助教智能体系统

基于 LangGraph 多智能体编排的 AI 助教平台，面向学生与教师提供 **智能问答、试卷批改、简历审查、模拟面试** 四大核心能力。系统采用 RAG 检索增强（Milvus 混合检索 + 重排序）与意图分类路由，将用户请求自动分发到对应的 Agent，并支持 SSE 流式输出。

## 核心功能

| 功能 | 说明 | 入口 |
|---|---|---|
| 💬 智能问答 | 基于课程知识库的 RAG 问答，混合检索（稠密 + 稀疏向量）+ BGE 精排，支持联网搜索兜底 | `/qa` |
| 📝 试卷批改 | 上传 Word 试卷，AI 自动批改打分，教师端审核确认后发布成绩 | `/exam` |
| 📄 简历审查 | 上传 PDF 简历，多维度（结构、内容、语言等）评分并生成改进报告 | `/resume` |
| 🎤 模拟面试 | 多阶段面试流程（暖场 → 技术 → 项目 → 收尾），实时追问并生成面试评估报告 | `/interview` |
| 🤖 统一对话 | 意图自动识别（本地 MiniLM 分类器），单 Agent 直达或跨 Agent Pipeline 串联（如"帮我改简历并准备面试"） | `/chat` |

## 系统架构

```
浏览器（Vue 3 + TypeScript + Element Plus，:3000）
   │ HTTP / SSE
   ▼
FastAPI 后端（:8000）
   ├── Orchestrator 编排层          ← 意图分类（MiniLM）→ 路由到 Agent / Pipeline
   │     ├── QA Agent               ← RAG 问答（BGE-M3 嵌入 + BGE-Reranker 精排）
   │     ├── Exam Agent             ← 试卷批改（python-docx 解析）
   │     ├── Resume Agent           ← 简历审查（PyMuPDF 解析）
   │     └── Interview Agent        ← 多阶段模拟面试
   ├── MCP 服务（/mcp/kb 知识库、/mcp/web-search 联网搜索）
   ├── LLM：DeepSeek（deepseek-chat / deepseek-coder，OpenAI 兼容接口）
   ├── 本地模型（进程内推理，无需独立服务）
   │     ├── all-MiniLM-L6-v2       意图分类
   │     ├── bge-m3                 文档嵌入（稠密 1024 维 + 稀疏双向量）
   │     └── bge-reranker-large     检索结果精排
   ├── PostgreSQL（用户、会话记忆、批改/审查记录，:5433）
   └── Milvus 向量库（知识库检索，:19531；etcd + MinIO 支撑，Attu 可视化管理 :30000）
```

**RAG 流水线**：文档加载（Markdown/PDF）→ 标题切分 + 字符切分（chunk 512 / overlap 100）→ 可选 Contextual RAG 上下文补充 → BGE-M3 双向量嵌入 → Milvus 入库；检索时混合召回 + BGE 精排。

**多租户**：所有数据按 `tenant_id` 隔离，默认租户 `tenant_default`。

## 技术栈

**后端**：Python 3.11+、FastAPI、LangChain 1.2 / LangGraph 1.0（版本锁定）、SSE 流式（sse-starlette）、MCP Python SDK、JWT 认证（python-jose + bcrypt）、SQLAlchemy 2 async + asyncpg、PyMilvus、pytest + pytest-asyncio

**前端**：Vue 3.4、TypeScript 5.4、Vite 5、Element Plus、Pinia、Vue Router、axios、markdown-it + highlight.js

**基础设施**：PostgreSQL 15、Milvus 2.4、etcd 3.5、MinIO、Docker Compose

## 目录结构

```
EduAgent/
├── backend/                      # FastAPI 后端
│   ├── main.py                   # 应用入口（lifespan：迁移 + 模型预热 + MCP 挂载）
│   ├── config.py                 # 配置中心（读取 .env.local）
│   ├── api/                      # 路由层（/api/v1/{auth,chat,qa,exam,resume,interview}）
│   ├── agents/                   # LangGraph 多智能体（qa / exam / resume / interview / reports）
│   ├── core/                     # 编排器、意图分类、知识库检索、重排序、记忆、LLM 工厂
│   ├── db/                       # 数据库迁移（启动时自动执行，幂等）
│   └── mcp/                      # MCP 服务器（知识库、Web 搜索）
├── frontend/                     # Vue 3 前端
│   └── src/                      # views（登录/仪表盘/QA/试卷/简历/面试）、components、stores
├── models/                       # 本地模型权重
│   ├── classifier/               # all-MiniLM-L6-v2（意图分类）
│   ├── embedding/                # bge-m3（文档嵌入）
│   └── reranker/                 # bge-reranker-large（精排）
├── scripts/                      # 运维脚本
│   ├── init_db.sql               # 建表脚本（PostgreSQL 容器首启自动执行）
│   ├── init_milvus.py            # 初始化 Milvus 向量集合
│   ├── build_knowledge_base.py   # 知识库构建（加载 → 切分 → 嵌入 → 入库）
│   └── seed_data.py              # 灌入测试账号
├── samples/                      # 样例文档（sample1.pdf、sample2.md）
├── tests/                        # pytest 测试（登录并发、SSE 等）
├── review/                       # 学习笔记 / demo 脚本（非核心代码）
├── docker-compose.yml            # PostgreSQL + etcd + MinIO + Milvus + Attu
├── requirements.txt              # Python 依赖（版本严格锁定）
└── .env.local.txt                # 环境变量模板（复制为 .env.local 后填写）
```

## 快速开始

### 0. 环境要求

- Python 3.11+（Windows 建议 conda；`backend/main.py` 已内置 conda DLL 路径兼容处理）
- Node.js 18+
- Docker Desktop（运行基础设施）

### 1. 安装后端依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.local.txt .env.local
```

编辑 `.env.local`，**必填三项**：

| 变量 | 说明 |
|---|---|
| `DB_PASSWORD` | PostgreSQL 密码（与 docker-compose 共用） |
| `DEEPSEEK_API_KEY` | DeepSeek API Key |
| `JWT_SECRET_KEY` | 一段随机字符串（JWT 签名密钥） |

其余保持默认即可：PostgreSQL `5433`、Milvus `19531`（与本机已有服务隔离）。

### 3. 启动基础设施（Docker）

```bash
docker-compose --env-file .env.local up -d
```

| 服务 | 端口 | 说明 |
|---|---|---|
| PostgreSQL | `5433` | 主数据库（首次启动自动执行 `scripts/init_db.sql` 建表） |
| Milvus | `19531` | 向量数据库 |
| Attu | `30000` | Milvus 可视化管理界面 |
| etcd / MinIO | 内部 | Milvus 依赖，不对外暴露 |

> 后端启动时会自动执行增量迁移（`backend/db/migrations.py`），建表脚本以容器首启执行的 `init_db.sql` 为准。

### 4. 准备本地模型

三个模型权重需放在 `models/` 对应目录（本项目已就位；如缺失请从 HuggingFace 下载）：

| 模型 | 目录 | 用途 |
|---|---|---|
| `all-MiniLM-L6-v2` | `models/classifier/` | 意图分类 |
| `bge-m3` | `models/embedding/` | 文档嵌入（稠密 + 稀疏） |
| `bge-reranker-large` | `models/reranker/` | 检索精排 |

后端启动时会并行预热这三个模型（首次加载较慢，预热避免首请求卡顿）。

### 5. 初始化向量库与测试数据

```bash
python scripts/init_milvus.py      # 创建 knowledge_domain 集合（稠密 + 稀疏双向量）
python scripts/seed_data.py        # 灌入测试账号
```

### 6. 构建课程知识库（可选，问答功能依赖）

编辑 `scripts/build_knowledge_base.py` 的 `__main__` 段，指定：

- `FILE_PATH`：课程文档路径（支持 `.md` / `.pdf`）
- `COURSE_ID`：课程 UUID
- `USE_CONTEXT`：是否启用 Contextual RAG（`False` 可快速调试，不消耗 API 配额）

```bash
python scripts/build_knowledge_base.py
```

### 7. 启动后端

在项目根目录执行：

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

- Swagger 文档：<http://localhost:8000/docs>
- 健康检查：<http://localhost:8000/health>

### 8. 启动前端

```bash
cd frontend
npm install
npm run dev        # http://localhost:3000
```

> ⚠️ **端口一致性**：前端默认把后端地址写成了 `8006`（Vite proxy 与 `frontend/.env.local` 的 `VITE_API_BASE_URL`）。若后端按第 7 步跑在 `8000`，请同步修改：
>
> - `frontend/vite.config.ts` → proxy target 改为 `http://localhost:8000`
> - `frontend/.env.local` → `VITE_API_BASE_URL=http://localhost:8000`（SSE 流式响应绕过 Vite proxy 直连后端，必须一致）

### 9. 测试账号

访问 <http://localhost:3000> 登录：

| 角色 | 邮箱 | 密码 |
|---|---|---|
| 管理员 | `admin@eduagent.local` | `Admin@123456` |
| 教师 | `teacher01@eduagent.local` | `Teacher@123456` |
| 学生 | `student01@eduagent.local` | `Student@123456` |
| 学生 | `student02@eduagent.local` | `Student@123456` |

## API 概览

统一前缀 `/api/v1`，认证方式：`Authorization: Bearer <JWT>`（有效期 7 天）。

| 模块 | 端点 | 说明 |
|---|---|---|
| 认证 | `POST /auth/login` | 登录获取 JWT |
| 认证 | `GET /auth/me` | 当前用户信息 |
| 统一对话 | `POST /chat/stream` | 统一入口，SSE 流式（意图路由 → Agent / Pipeline） |
| 智能问答 | `POST /qa/chat` / `POST /qa/chat/stream` | 知识库问答（普通 / SSE 流式） |
| 智能问答 | `GET /qa/sessions/{session_id}/history` | 会话历史 |
| 简历审查 | `POST /resume/upload` | 上传 PDF 简历（202，异步处理） |
| 简历审查 | `GET /resume/reviews` / `GET /resume/reviews/{review_id}` | 审查报告列表 / 详情 |
| 简历审查 | `DELETE /resume/reviews/{review_id}` | 删除审查记录 |
| 试卷批改 | `POST /exam/submit` | 提交 Word 试卷（202，异步批改） |
| 试卷批改 | `GET /exam/my-submissions` / `GET /exam/my-submissions/{submission_id}` | 我的提交 / 批改结果 |
| 试卷批改 | `GET /exam/pending-reviews` | 教师端：待审核列表 |
| 试卷批改 | `POST /exam/submissions/{submission_id}/confirm` | 教师端：确认批改结果 |
| 模拟面试 | `POST /interview/sessions` | 创建面试会话 |
| 模拟面试 | `POST /interview/sessions/{session_id}/chat` / `.../chat/stream` | 面试对话（普通 / SSE 流式） |
| 模拟面试 | `GET /interview/sessions/{session_id}/report` | 面试评估报告 |
| 模拟面试 | `GET /interview/sessions` | 面试会话列表 |
| MCP | `/mcp/kb` | 知识库 MCP Server（streamable HTTP） |
| MCP | `/mcp/web-search` | 联网搜索 MCP Server（DuckDuckGo，可用 Tavily 替换） |

## 测试

```bash
pytest              # 按根目录 pytest.ini 配置运行 tests/ 目录（asyncio 自动模式）
```

## 注意事项

- **依赖版本锁定**：`requirements.txt` 中 LangChain / LangGraph 系列版本严格锁定，请勿随意升级。
- **bcrypt 兼容**：`passlib 1.7.4` 与 `bcrypt>=5` 不兼容，必须锁定 `bcrypt==4.0.1`。
- **Web 搜索**：默认使用免费的 DuckDuckGo（无需 Key）；配置 `TAVILY_API_KEY` 可切换 Tavily。
- **使用率/限额**：DeepSeek API Key 会话按量计费；Contextual RAG 会额外消耗 API 配额，可用 `USE_CONTEXT=False` 跳过。
- **数据库初始化**：修改表结构时更新 `scripts/init_db.sql`（容器首启执行）与 `backend/db/migrations.py`（后端启动幂等迁移），保证全新环境可一键启动。
