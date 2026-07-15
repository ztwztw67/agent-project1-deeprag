FROM ubuntu:latest
LABEL authors="郑天维"

ENTRYPOINT ["top", "-b"]

# ========== 第一阶段：安装依赖 ==========
FROM python:3.11-slim AS builder

WORKDIR /app

# 先复制依赖清单（利用 Docker 缓存层——依赖不变就不重新装）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ========== 第二阶段：运行应用 ==========
FROM python:3.11-slim

WORKDIR /app

# 从第一阶段把装好的依赖复制过来（不需要重新 pip install）
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# 复制项目代码
COPY . .

# 暴露端口（文档作用，实际端口在 docker-compose 或命令行指定）
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]