"""
应用配置管理
============
使用 pydantic-settings 从 .env 文件和环境变量加载配置。
比 os.getenv() 好在：类型校验、默认值、IDE 自动补全。
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """全局配置类 —— 项目中所有配置从这里取"""

    # model_config 告诉 pydantic-settings：
    # "去 .env 文件里找这些变量的值，找不到就用默认值"
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # ========== 应用基础 ==========
    app_name: str = "AgentProject"
    app_env: str = "development"  # development / production / test
    debug: bool = True

    # ========== 数据库 ==========
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "your_password"
    mysql_database: str = "agent_db"

    # property 装饰器：不存新字段，而是根据已有字段"算"出结果
    # 调用 settings.database_url 时动态拼接
    @property
    def database_url(self) -> str:
        """拼接 SQLAlchemy 用的数据库连接字符串"""
        return (
            f"mysql+aiomysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )

    # ========== Redis ==========
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None

    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}"
        return f"redis://{self.redis_host}:{self.redis_port}"

    # ========== JWT ==========
    jwt_secret_key: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24 小时

    # ========== LLM ==========
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    llm_reasoning_effort: str = "high"  # v4-pro 推理深度：low / medium / high
    llm_thinking: str = "enabled"  # enabled = 开启思维链，disabled = 直出答案


# 全局唯一实例 —— 其他模块统一 from backend.config import settings
settings = Settings()