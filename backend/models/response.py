from pydantic import BaseModel
from typing import Optional, Any, Generic, TypeVar

T = TypeVar("T")  # 泛型占位符，允许 APIResponse 包裹任意类型的数据


class APIResponse(BaseModel, Generic[T]):
    """标准成功响应
    用法：
        return APIResponse(message="成功", data=user_obj)
        return APIResponse(code=400, message="参数错误")
    """
    code: int = 200           # 业务状态码，200 表示成功
    message: str = "success"  # 给前端看的提示信息
    data: Optional[T] = None  # 实际数据，可以是任意类型


class ErrorResponse(BaseModel):
    """错误响应（一般配合 HTTPException 使用）"""
    code: int
    message: str
    detail: Optional[str] = None