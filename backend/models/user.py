
"""
用户相关数据模型
===============
注册、登录、返回用户信息 —— 每种场景用不同的模型，
因为"注册时需要的字段"和"返回给前端的字段"不一样。
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime
import re


class UserRegisterRequest(BaseModel):
    """注册请求 —— 用户提交的注册表单"""
    username: str = Field(
        ...,                    # ... 表示必填
        min_length=3,           # FastAPI 自动校验，不符合直接 422
        max_length=32,
        description="用户名",
        examples=["zhangsan"],
    )
    password: str = Field(
        ...,
        min_length=6,
        max_length=64,
        description="密码，最少 6 位",
    )
    email: Optional[str] = Field(
        None,
        pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
        description="邮箱（可选，但如果填了格式必须正确）",
    )

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        """自定义校验：用户名只允许字母、数字、下划线"""
        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError("用户名只能包含字母、数字、下划线")
        return v


class UserLoginRequest(BaseModel):
    """登录请求"""
    username: str = Field(..., min_length=3)
    password: str = Field(..., min_length=6)


class UserResponse(BaseModel):
    """返回给前端的用户信息
    ⚠️ 绝对不能包含 password 字段！"""
    id: int
    username: str
    email: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True  # 允许从 SQLAlchemy ORM 对象自动转换


class TokenResponse(BaseModel):
    """登录成功后返回的 Token"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # 过期时间（秒）