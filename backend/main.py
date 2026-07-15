"""
FastAPI 应用入口
===============
整个后端从这里启动。所有路由、中间件、生命周期事件在这里注册。
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.routers import auth
from backend.middleware.log import LogMiddleware


# lifespan：替代旧版 on_event("startup") / on_event("shutdown")
# 所有启动/关闭逻辑集中在一起
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ===== 启动时执行 =====
    print(f"🚀 {settings.app_name} 启动中...")
    print(f"   环境: {settings.app_env}")
    print(f"   LLM: {settings.llm_model} @ {settings.openai_base_url}")
    yield  # ← 应用在这里运行
    # ===== 关闭时执行 =====
    print("👋 正在关闭...")


# 创建 FastAPI 应用 —— 这是整个后端的"大脑"
app = FastAPI(
    title=settings.app_name,
    description="Agent 应用开发 —— 秋招冲刺项目骨架",
    version="0.1.0",
    lifespan=lifespan,
)

# ===== 中间件注册（按添加顺序执行） =====

# 1. CORS 中间件 —— 允许浏览器跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 开发阶段允许所有来源，生产要限制
    allow_credentials=True,
    allow_methods=["*"],       # GET, POST, PUT, DELETE... 全允许
    allow_headers=["*"],
)

# 2. 请求日志中间件
app.add_middleware(LogMiddleware)

# ===== 路由注册 =====
# prefix="/auth" 意味着 auth.router 里的所有路径都自动加上 /auth
# 比如 @router.post("/register") 的实际访问路径是 POST /auth/register
app.include_router(auth.router, prefix="/auth", tags=["认证"])

# 后续添加新模块只需加一行：
# app.include_router(chat.router, prefix="/chat", tags=["聊天"])
# app.include_router(rag.router, prefix="/rag", tags=["RAG"])


# ===== 基础接口 =====

@app.get("/")
async def root():
    """根路径 —— 用来确认服务还活着"""
    return {
        "app": settings.app_name,
        "version": "0.1.0",
        "docs": "/docs",  # FastAPI 自动生成的 Swagger 文档地址
    }


@app.get("/health")
async def health():
    """健康检查 —— Docker/K8s 定期访问这个接口判断服务是否存活"""
    return {"status": "healthy"}