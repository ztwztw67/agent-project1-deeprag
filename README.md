# DeepRAG · 多源检索增强生成智能问答 Agent

> 一个基于 **FastAPI + LangChain/LangGraph** 构建的检索增强生成（RAG）系统，历经四版迭代，从"单文档向量检索"演进到"多源 Agent 路由 + 混合检索 + 重排序"的完整方案，并配套离线评估体系量化每一次改进。

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.139-009688.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.x-1C3C3C.svg)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.x-333.svg)](https://langchain-ai.github.io/langgraph/)

---

## 📖 项目简介

企业知识库问答的痛点在于：**专有名词检索不准、口语化提问检索失败、单一向量检索遗漏关键词精确匹配**。本项目围绕这三个问题，实现了一条可降级、可评估的 RAG 检索链路：

```
用户提问 → Query 重写 → 混合检索（向量 + BM25）→ Cross-Encoder 重排序 → Top-3 → LLM 生成
```

在此基础上，用 ReAct Agent 把本地知识库、Web 搜索、数据库查询三类检索源封装成工具，让 LLM 自主决定"该去哪里找答案"。

> 本项目为 **Agent 应用开发 · 秋招冲刺项目** 的实践项目，代码注释保留了大量面试导向的设计说明。

## ✨ 核心特性

- **四版迭代的检索链路**：纯向量 → 多源 Agent → 混合检索 + Query 重写 + Rerank → SSE 流式 + 评估
- **混合检索（Hybrid Search）**：Dense（BGE 语义）+ Sparse（BM25 关键词）加权融合，兼顾语义与精确匹配
- **Cross-Encoder 重排序**：Bi-Encoder 粗筛、Cross-Encoder 精排的分工架构，显著提升 Top-K 命中排名
- **多源 Agent 路由**：基于 LangGraph 的 ReAct Agent，自主调度本地检索 / Web 搜索 / Text-to-SQL 三个工具
- **SSE 流式输出**：token 级实时推送 + 工具调用状态 + 参考来源，前端可感知检索过程
- **离线评估体系**：chunk-ID 确定性标注 + Hit Rate / MRR 双指标，让每次改进可量化
- **可降级设计**：知识库为空自动回退、Query 重写失败回退原词、LLM 调用指数退避重试

## 🧭 RAG 检索链路演进

| 版本 | 核心能力 | 解决的问题 |
|------|---------|-----------|
| **v1** | 单文档 RAG：切分 + 向量化 + 相似度检索 + 生成 | 跑通基础链路 |
| **v2** | 多源 Agent 路由：ReAct Agent + 3 个检索工具 | 单知识库覆盖不足，需自主选择数据源 |
| **v3** | 混合检索 + Query 重写 + Cross-Encoder 重排序 | 口语化提问检索失败、关键词精确匹配遗漏 |
| **v4** | SSE 流式输出 + chunk-ID 评估体系 | 交互体验差、改进无法量化验证 |

### 检索链路的两个关键设计

**① 规则前置过滤的 Query 重写**（`_should_rewrite`）

不是所有 query 都送去 LLM 重写——改写本身会引入约 1s 延迟和潜在偏差。先经轻量规则过滤：太短跳过、含技术术语跳过、含时间限定词跳过，只放行"明显口语化/模糊"的 query。

**② 可降级的检索器组装**（`search_v3`）

```
rewrite_query(query)  → 多个精确检索词
   ↓ 每个检索词
final_retriever = ContextualCompressionRetriever(
    base_retriever = EnsembleRetriever(Dense 0.6 + BM25 0.4),  # 粗筛
    base_compressor = CrossEncoderReranker(top_n=3)              # 精排
)
```

知识库为空时 `final_retriever` 为 `None`，自动回退到 v1 纯向量检索——**链路每一环都可降级，不因单点失败阻塞主流程**。

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| Web 框架 | FastAPI 0.139 + Uvicorn |
| LLM 框架 | LangChain 0.3.x + LangGraph 0.2.x |
| 向量数据库 | ChromaDB 0.6 |
| Embedding | BGE-small-zh-v1.5（Bi-Encoder，本地推理） |
| Rerank | BGE-reranker-base（Cross-Encoder，本地推理） |
| LLM | DeepSeek（OpenAI 兼容协议，可替换） |
| 关键词检索 | BM25（rank-bm25） |
| Web 搜索 | Tavily Search API |
| 数据库 | MySQL（SQLAlchemy 2.0 异步）+ Redis |
| 认证 | JWT（python-jose）+ bcrypt |
| 文档解析 | PyMuPDF（PDF）/ 原生 txt |
| 配置管理 | pydantic-settings |
| 部署 | Docker + Docker Compose |

## 📁 项目结构

```
agent-deeprag/
├── backend/
│   ├── main.py              # FastAPI 入口：中间件、路由注册、lifespan
│   ├── config.py            # pydantic-settings 配置（从 .env 读取）
│   ├── db.py                # SQLAlchemy 2.0 异步引擎 + 依赖注入
│   ├── routers/
│   │   ├── auth.py          # 注册 / 登录 / 当前用户（JWT）
│   │   └── rag.py           # 文档上传 / Agent 问答 / SSE 流式
│   ├── services/
│   │   ├── rag_service.py   # 核心 RAG 链路：切分、向量化、检索、生成
│   │   └── agent_tools.py   # 3 个检索工具 + ReAct Agent
│   ├── evaluation/
│   │   └── eval_rag.py      # 评估：测试集 + Hit Rate / MRR
│   ├── middleware/
│   │   └── log.py           # 请求日志中间件
│   └── models/              # Pydantic 请求/响应模型
├── models/                  # 本地模型（BGE），需手动下载，见下文
├── chroma_db/               # Chroma 向量持久化目录
├── requirements.txt         # Python 依赖
├── Dockerfile               # 多阶段构建镜像
├── docker-compose.yml       # FastAPI + MySQL + Redis 编排
├── .env.example             # 环境变量模板
└── text-agent.py            # 早期单文件原型
```

## 🚀 快速开始

### 环境要求

- Python 3.11
- （可选）Docker + Docker Compose

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> ⚠️ `passlib 1.7.4` 与 `bcrypt>=4.1` 存在兼容问题。若安装或运行时报 bcrypt 相关错误，请将 `requirements.txt` 中的 `bcrypt==5.0.0` 降级为 `bcrypt==4.0.1`。

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，至少填入：

```ini
OPENAI_API_KEY=sk-your-api-key-here   # DeepSeek / OpenAI 兼容接口的 Key
OPENAI_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
TAVILY_API_KEY=tvly-your-key-here      # 使用 Web 搜索工具时需要
JWT_SECRET_KEY=换成一个随机字符串
```

### 3. 下载本地模型

`models/` 目录（约 3GB）未纳入版本控制，需手动下载两个模型：

```bash
# Embedding 模型（约 100MB）
huggingface-cli download BAAI/bge-small-zh-v1.5 --local-dir models/bge-small-zh-v1.5

# Rerank 模型（约 400MB）
huggingface-cli download BAAI/bge-reranker-base --local-dir models/bge-reranker-base
```

> 国内网络可用 ModelScope 替代：
> `modelscope download --model BAAI/bge-reranker-base --local_dir models/bge-reranker-base`

下载后确认 `backend/services/rag_service.py` 中的 `model_name` 路径与你本机的 `models/` 目录一致（当前硬编码为开发机的绝对路径，需按需修改）。

### 4. 启动服务

```bash
uvicorn backend.main:app --reload
```

访问 <http://localhost:8000/docs> 查看 Swagger 接口文档。

## 📡 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 服务信息 |
| GET | `/health` | 健康检查（Docker/K8s 探活） |
| POST | `/auth/register` | 用户注册 |
| POST | `/auth/login` | 用户登录，返回 JWT |
| GET | `/auth/me` | 当前用户（未实现，预留） |
| POST | `/rag/upload` | 上传文档（txt / pdf），切分 + 向量化入库 |
| POST | `/rag/chat` | Agent 问答，返回答案 + 参考来源 |
| POST | `/rag/chat/stream` | SSE 流式问答，实时推送 token / 工具状态 |

### 使用示例

```bash
# 1. 上传文档
curl -F "file=@test_doc.txt" http://localhost:8000/rag/upload

# 2. 问答
curl -X POST http://localhost:8000/rag/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "云枢科技的 CTO 是谁？"}'
```

### SSE 流式事件类型

`/rag/chat/stream` 返回 `text/event-stream`，前端根据 `type` 字段切换 UI 状态：

| type | 含义 |
|------|------|
| `token` | LLM 逐字输出 |
| `tool_start` | Agent 开始调用工具 |
| `tool_result` | 工具返回结果摘要（参考来源） |
| `error` | 检索或生成失败 |
| `done` | 流结束 |

## 🧪 评估体系

离线评估解决了"改完检索链路，怎么证明它变好了"的问题。核心思路是**确定性标注 + 双指标量化**：

- **标注**：上传文档后运行 `show_chunks()` 查出每个 chunk 的稳定 ID（`文件MD5前8位-序号`），人工标注哪些 chunk 包含正确答案；未标注时回退到关键词子串匹配。
- **指标**：
  - **Hit Rate**：正确答案出现在 Top-3 结果中的 query 占比
  - **MRR**（Mean Reciprocal Rank）：正确答案首次出现排名的倒数平均值，反映"排得是否靠前"
- **测试集**：覆盖基础召回、表格/易混实体、否定/例外、时效冲突、诚实性（文档无答案）等 8 类场景。

```bash
python -m backend.evaluation.eval_rag
```

**示例结果**（v1 纯向量 vs v3 混合 + Rerank）：

| 版本 | Hit Rate | MRR |
|------|----------|-----|
| v1 纯向量检索 | — | 0.537 |
| v3 混合 + Rerank | 77.78% | **0.778** |

> v3 通过"混合检索 + 重排序"，不仅召回了更多正确答案，还把它们排到了更靠前的位置——MRR 提升 45%。实际数值以本机运行结果为准。

## 🐳 Docker 部署

```bash
docker compose up -d --build
```

编排包含三个服务：`api`（FastAPI，挂载 `models/` 与 `chroma_db/` 卷）、`mysql`、`redis`。

## 📌 已知限制

- **模型路径硬编码**：`rag_service.py` 中的 `model_name` 是开发机绝对路径，多机部署需改为环境变量或相对路径。
- **模块级初始化**：模型在 `import` 时加载，单元测试难以 mock，生产建议改为懒加载工厂 + 依赖注入。
- **同步阻塞检索**：`upload` / `search` / `generate` 为同步操作，Chroma 0.6 暂不支持异步，高并发场景可换 Milvus / Qdrant。
- **认证为内存实现**：`/auth` 目前用内存字典模拟，数据库层（SQLAlchemy）已搭框架但未真正接入 MySQL。

## 📄 License

待定。
