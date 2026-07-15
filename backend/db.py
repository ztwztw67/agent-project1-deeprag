"""
数据库连接管理
=============
用 SQLAlchemy 2.0 异步引擎连接 MySQL。
项目初期先搭好框架，等需要连数据库时直接复用。
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from backend.config import settings


# Step 1：创建异步引擎 —— 负责和 MySQL 通信
# echo=True 会在终端打印所有 SQL 语句，开发时非常有用
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,  # 开发环境打印 SQL，生产环境关闭
    pool_size=10,         # 连接池大小
    max_overflow=20,      # 超出 pool_size 时最多再开几个连接
)

# Step 2：创建会话工厂 —— 每次请求从这里"借"一个数据库会话
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # commit 后不使对象过期
)


# Step 3：声明基类 —— 所有 ORM 模型继承自它
class Base(DeclarativeBase):
    pass


# Step 4：FastAPI 依赖注入 —— 每个请求自动获取/释放数据库会话
async def get_db():
    """FastAPI 依赖：在请求开始时创建会话，结束时自动关闭

    用法：
        @app.get("/users")
        async def list_users(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(User))
            return result.scalars().all()
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session        # 把会话交给路由函数
            await session.commit()  # 如果没有异常，提交事务
        except Exception:
            await session.rollback()  # 有异常就回滚
            raise
        # with 块结束后自动 close，不需要手动处理