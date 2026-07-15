"""
认证路由：注册、登录
===================
"""
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException
from jose import jwt  # python-jose 库，生成和校验 JWT

from backend.config import settings
from backend.models.user import (
    UserRegisterRequest,
    UserLoginRequest,
    UserResponse,
    TokenResponse,
)
from backend.models.response import APIResponse, ErrorResponse

router = APIRouter()

# ═══════════════════════════════════════════════════════════
# v1 模拟版：用内存字典存储用户（不依赖 MySQL，先跑通）
# ═══════════════════════════════════════════════════════════

# 模拟"数据库"——就是一个字典，服务重启后数据丢失
_fake_db: dict[str, dict] = {}


def _hash_password(password: str) -> str:
    """密码哈希 —— 使用 passlib 的 bcrypt 算法（不可逆）

    ⚠️ 为什么不能存明文密码：
    数据库一旦泄露 → 所有用户的密码被看光。
    bcrypt 是单向哈希 → 即使数据库泄露，攻击者也推不出原始密码。
    """
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return pwd_context.hash(password)


def _verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码：对用户输入的密码做哈希，和数据库里的哈希值比对"""
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return pwd_context.verify(plain_password, hashed_password)


def _create_token(user_id: int, username: str) -> str:
    """生成 JWT Token

    JWT 原理（通俗版）：
    服务器用密钥签了一个"通行证"，上面写着 user_id + 过期时间。
    后续请求带着这个通行证，服务器验签名、看有没有过期。
    因为签名只有服务器能生成，所以无法伪造。
    """
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {
        "sub": str(user_id),          # subject — 令牌的主人
        "username": username,
        "exp": expire,                # expiration — 过期时间
        "iat": datetime.now(timezone.utc),  # issued at — 签发时间
    }
    token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return token


@router.post("/register", response_model=APIResponse[UserResponse])
async def register(req: UserRegisterRequest):
    """用户注册

    流程：检查用户名 → 密码哈希 → 存入"数据库" → 返回用户信息
    """
    # 1. 检查用户名是否已存在
    if req.username in _fake_db:
        raise HTTPException(status_code=409, detail="用户名已被占用")

    # 2. 创建用户（密码做哈希，不存明文）
    now = datetime.now(timezone.utc)
    user_dict = {
        "id": len(_fake_db) + 1,
        "username": req.username,
        "hashed_password": _hash_password(req.password),
        "email": req.email,
        "created_at": now,
    }
    _fake_db[req.username] = user_dict

    # 3. 返回用户信息（不含密码！）
    return APIResponse(
        message="注册成功",
        data=UserResponse(
            id=user_dict["id"],
            username=user_dict["username"],
            email=user_dict["email"],
            created_at=user_dict["created_at"],
        ),
    )


@router.post("/login", response_model=APIResponse[TokenResponse])
async def login(req: UserLoginRequest):
    """用户登录

    流程：查用户是否存在 → 验证密码 → 生成 JWT → 返回 Token
    """
    # 1. 查用户
    user_dict = _fake_db.get(req.username)
    if not user_dict:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    # 2. 验密码
    if not _verify_password(req.password, user_dict["hashed_password"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    # 3. 生成 Token
    token = _create_token(user_dict["id"], user_dict["username"])

    return APIResponse(
        message="登录成功",
        data=TokenResponse(
            access_token=token,
            expires_in=settings.jwt_expire_minutes * 60,
        ),
    )


@router.get("/me", response_model=APIResponse[UserResponse])
async def get_current_user():
    """获取当前登录用户（需要 JWT Token 验证——后续项目补上）"""
    # TODO: 从请求头解析 Token → 验签 → 查用户
    raise HTTPException(status_code=501, detail="尚未实现——Week 2 补上")