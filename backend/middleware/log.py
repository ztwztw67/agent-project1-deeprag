"""
请求日志中间件
=============
记录每个 HTTP 请求的：方法、路径、耗时、状态码。
这是后端调试最基本的手段——请求进来了吗？哪个请求慢？
"""
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# 配置日志格式：时间 + 级别 + 消息
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agent-project")


class LogMiddleware(BaseHTTPMiddleware):
    """记录每个请求的方法、路径和耗时"""

    async def dispatch(self, request: Request, call_next):
        # 请求进来时记录时间
        start_time = time.time()

        # call_next 把请求传给下一个中间件/路由，等它处理完
        response = await call_next(request)

        # 请求处理完，算耗时
        duration = (time.time() - start_time) * 1000  # 转毫秒

        # 一行日志包含所有关键信息
        logger.info(
            f"{request.method} {request.url.path} "
            f"→ {response.status_code} "
            f"({duration:.0f}ms)"
        )
        return response