"""RAG 聊天路由 —— v1 单文档 RAG"""
import os
import tempfile

from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel, Field
from backend.models.response import APIResponse
from backend.services.rag_service import search, generate, upload_document,EmbeddingError
from backend.services.agent_tools import agent
from langchain_core.messages import HumanMessage
from fastapi.responses import StreamingResponse
import logging
import json

logger = logging.getLogger("agent-project")

router = APIRouter()


class RagChatRequest(BaseModel):
    """RAG 问答请求体"""
    query: str = Field(..., description="用户问题", min_length=1, examples=["什么是RAG？"])


class RagChatResponse(BaseModel):
    """RAG 问答响应 data 字段"""
    query: str
    answer: str
    sources: list[str]


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    """上传文档（支持 txt / pdf）→ 切分 → 向量化 → 存入 Chroma"""
    content = await file.read()
    original_filename = file.filename or "upload"

    # 保留原始扩展名，upload_document 根据扩展名判断解析方式
    suffix = os.path.splitext(original_filename)[1] or ".txt"
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=suffix, delete=False
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunk_count = upload_document(tmp_path)
        return APIResponse(
            message=f"文档 {original_filename} 已处理完成",
            data={"filename": original_filename, "chunks": chunk_count},
        )
    except EmbeddingError as e:
        # ⚠️ upload_document 内部可能因 PDF 解析失败/切分异常/向量化失败等原因
        # 抛出 EmbeddingError——这里捕获后返回结构化错误，而非让 FastAPI 兜底返回 500
        return APIResponse(code=500, message=f"文档处理失败: {str(e)}")
    finally:
        os.unlink(tmp_path)  # 清理临时文件

def _extract_sources(messages: list) -> list[str]:
    """从 Agent 消息历史中提取检索来源。

    ReAct Agent 的消息流：
      HumanMessage（用户问题）
      → AIMessage(tool_calls) （Agent 决定调用工具）
      → ToolMessage （工具返回结果）
      → AIMessage （最终回答）

    本函数收集所有 ToolMessage 的内容作为引用来源，供前端展示「参考了哪些内容」。
    """
    sources = []
    for msg in messages:
        # ToolMessage 有 tool_call_id 属性，HumanMessage/AIMessage 没有
        if hasattr(msg, "tool_call_id"):
            content = msg.content
            if isinstance(content, str) and content.strip():
                sources.append(content[:300] + ("..." if len(content) > 300 else ""))
    return sources

# ⚠️ 定义 Pydantic 请求体模型——v1 开始就养成习惯，后续 v2/v4 直接复用
class RagChatRequest(BaseModel):
    query: str = Field(..., description="用户问题", min_length=1, examples=["什么是RAG？"])

@router.post("/chat", response_model=APIResponse[RagChatResponse])
async def rag_chat(req: RagChatRequest):
    """v2 Agent 问答 —— LLM 自主选择检索源"""
    if not req.query.strip():
        return APIResponse(code=400, message="query 不能为空")

    try:
        result = agent.invoke({"messages": [HumanMessage(content=req.query)]})
        # langgraph agent 返回 {"messages": [...]}，最后一条 AIMessage 是最终回答
        # ⚠️ 如果 Agent 中途 tool 调用全部失败，最后一条可能是 ToolMessage
        # 生产环境应遍历 messages 找最后一条 AIMessage
        answer = result["messages"][-1].content
        # 从 Agent 的消息历史中提取检索来源
        sources = _extract_sources(result["messages"])
    except Exception as e:
        logger.error("Agent 调用失败 (query=%s): %s", req.query[:50], e)
        return APIResponse(code=500, message=f"Agent 执行失败: {str(e)}")

    return APIResponse(
        message="ok",
        data={
            "query": req.query,
            "answer": answer,
            "sources": sources,
        },
    )

def _collect_sources(messages: list) -> list[str]:
     """从 Agent 消息历史中提取工具调用结果作为参考来源。

     SSE 流式端点无法像 /chat 那样在 response body 中返回 sources，
     因此改为在前端侧收集——Agent 完成工具调用后，通过 tool_result 事件
     将每条检索结果的摘要推给前端，前端累积展示"参考来源"。
     """
     sources = []
     for msg in messages:
         if hasattr(msg, "tool_call_id") and hasattr(msg, "content"):
             content = msg.content
             if isinstance(content, str) and content.strip():
                 sources.append(content[:200] + ("..." if len(content) > 200 else ""))
     return sources


@router.post("/chat/stream")
async def chat_stream(req: RagChatRequest):
    """SSE 流式 Agent 问答 —— 实时推送 token + 工具调用状态 + 参考来源

    事件类型（前端根据 type 字段切换 UI 状态）：
      token       — LLM 逐字输出，前端追加到对话框
      tool_start  — Agent 开始调用工具，前端显示"正在搜索知识库..."
      tool_result — 工具返回结果摘要，前端累积到"参考来源"列表
      error       — 检索或 LLM 调用失败，前端显示错误提示
      done        — 流结束
    """
    async def generate():
        tool_results = []  # 收集本轮对话中所有工具调用结果
        try:
            async for event in agent.astream_events(
                {"messages": [HumanMessage(content=req.query)]},
                version="v2",
            ):
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:  # 部分 chunk 的 content 为空字符串
                        # yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                        yield f"data: {json.dumps({'type': 'token', 'content': content}, ensure_ascii=False)}\n\n"

                elif kind == "on_tool_start":
                    tool_name = event["name"]
                    tool_results.append({"tool": tool_name, "status": "running"})
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': tool_name})}\n\n"

                elif kind == "on_tool_end":
                    # 工具执行完成，推送结果摘要供前端展示参考来源
                    output = event["data"].get("output", "")
                    if isinstance(output, str) and output.strip():
                        summary = output[:300] + ("..." if len(output) > 300 else "")
                        tool_results[-1]["summary"] = summary if tool_results else None
                        yield f"data: {json.dumps({'type': 'tool_result', 'content': summary})}\n\n"

            # 流正常结束
            yield "data: {}\n\n".format(json.dumps({"type": "done"}))

        except Exception as e:
            # ⚠️ 企业级 SSE：异常必须作为结构化事件推给前端，
            # 不能让生成器静默崩溃——否则前端看到的是"连接断开"而无任何错误提示
            logger.error("SSE 流式生成失败 (query=%s): %s", req.query[:50], e)
            yield f"data: {json.dumps({'type': 'error', 'message': f'生成失败: {str(e)}'})}\n\n"
            yield "data: {}\n\n".format(json.dumps({"type": "done"}))

    return StreamingResponse(generate(), media_type="text/event-stream; charset=utf-8")
