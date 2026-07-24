"""RAG 核心服务 —— Project 1 的检索增强生成引擎

v1：单文档 RAG —— 切分 + 向量化 + 检索 + 生成
v2 见 agent_tools.py，v3/v4 将继续在本文件追加

----
设计说明（面试用）：
1. 模块级变量（text_splitter / embedding_model / vectorstore / llm）在 import 时初始化
   → 优点：v2 的 agent_tools.py 可以直接 import，代码简洁
   → 缺点：import 本模块会触发模型下载，单元测试无法 mock
   → 生产改进：用工厂函数/依赖注入延迟初始化（见文件末尾注释）
2. LLM 调用加了指数退避重试——应对 API 限流和瞬时网络故障
3. 自定义异常类让调用方能区分"检索失败"和"生成失败"
"""
import hashlib
import logging
import os
import time
from openai import OpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
# v3 最终检索链路：Query → rewrite → ensemble(Dense+BM25) → Rerank → Top-3 → LLM
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker  # ⚠️ 在 langchain 主包，不在 community
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


from backend.config import settings

logger = logging.getLogger("agent-project")

# ============================================================
# 自定义异常 —— 让调用方知道哪一环节出错
# ============================================================


class RAGException(Exception):
    """RAG 链路异常基类"""


class EmbeddingError(RAGException):
    """向量化失败"""


class RetrievalError(RAGException):
    """检索失败"""


class GenerationError(RAGException):
    """LLM 生成失败（含重试耗尽）"""


# ============================================================
# 基础设施：模型初始化（模块加载时执行一次）
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 中文约 500 字一个 chunk
    chunk_overlap=50,    # 10% overlap 保持上下文
    separators=["\n\n", "\n", "。", "！", "？", "，", " "]  # 按优先级切
)

embedding_model = HuggingFaceBgeEmbeddings(
    model_name="F:/python-project/agent-deeprag/models/bge-small-zh-v1.5",
    # ⚠️ 使用本地下载的 bge-small 模型 (~100MB)，避免 Hugging Face 在线下载 504 超时
    # bge-large (~1.3GB) 需要更大显存，可从 ModelScope 下载后替换路径
    model_kwargs={"device": "cpu"},   # CPU 推理（当前 PyTorch 为 CPU 版本）
    encode_kwargs={"normalize_embeddings": True}  # 归一化，余弦相似度 = 内积
)

vectorstore = Chroma(
    collection_name="knowledge_base",
    embedding_function=embedding_model,
    persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
)

# # Dense Retriever（语义）
# dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
#
# # Sparse Retriever（关键词，BM25 算法）
# # ⚠️ all_chunks 来自 v1 的 rag_service.py（每次 upload_document 时追加）
# # ⚠️ all_chunks 是 list[str]，不是 list[Document]——所以用 from_texts 而非 from_documents
# # bm25_retriever = BM25Retriever.from_texts(all_chunks, k=5)
#
# # Ensemble：加权融合
# ensemble = EnsembleRetriever(
#     retrievers=[dense_retriever, bm25_retriever],
#     weights=[0.6, 0.4]  # 语义权重更高，但保留关键词兜底
# )

# ⚠️ Reranker 是独立的 Cross-Encoder 模型，与 v1 的 bge-small-zh（Bi-Encoder，做向量化）
# 架构不同、权重不通用，无法复用，需要单独下载。
# 和 bge-small 同样的理由（HF 在线下载易 504、默认缓存到 C 盘 ~/.cache/huggingface），
# 先从 ModelScope 下载到本地 models/ 目录再用本地路径：
#   modelscope download --model BAAI/bge-reranker-base --local_dir F:/python-project/agent-deeprag/models/bge-reranker-base
reranker_model = HuggingFaceCrossEncoder(
    model_name="F:/python-project/agent-deeprag/models/bge-reranker-base",
    # 选型：bge-reranker-large(~1.3GB) 精度更高，是中文 rerank 首选；
    # 但当前 PyTorch 为 CPU 版，Cross-Encoder 每条候选都要完整前向一次（rerank 10 条 = 10 次前向），
    # large 在 CPU 上单次查询可能 10s+ —— base(~400MB) 是 CPU 场景的务实选择。
    # 换 GPU 后升级：把本行路径与上面下载命令中的 base 换成 large 即可。
    model_kwargs={"device": "cpu"},
)

# Bi-Encoder（粗筛）是 retriever，Cross-Encoder（精排）是 compressor
# 分工：retriever 从海量文档中捞候选 → compressor 对候选精排
compressor = CrossEncoderReranker(
    model=reranker_model,  # ⚠️ langchain 0.3.x 要求传 BaseCrossEncoder 实例，传模型名字符串会 ValidationError
    top_n=3                # 最终只保留 Top-3 最相关的给 LLM
)
# final_retriever = ContextualCompressionRetriever(
#     base_retriever=ensemble,       # ensemble = Dense + BM25 混合，做粗筛
#     base_compressor=compressor     # Cross-Encoder，做精排
# )

# ⚠️ v2 的 agent_tools.py 会 import 这里的 llm 和 vectorstore
# from backend.services.rag_service import vectorstore, llm
llm = OpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url,
)

# ============================================================
# v1 核心函数
# ============================================================

# ⚠️ v3 的 BM25Retriever 需要所有 chunk 文本做关键词索引
# 每次 upload_document() 把新 chunk 追加到此处
all_chunks: list[str] = []

# ====== v3 检索器组装（必须在 all_chunks 定义之后） ======
ensemble = None          # Dense + BM25 混合检索器，由 refresh_retrievers() 构建
final_retriever = None   # 混合 + Rerank 完整链路，同上


def refresh_retrievers() -> None:
    """构建/重建 v3 检索链路。启动时调用一次，之后每次 upload_document 后调用。"""
    global ensemble, final_retriever

    if not all_chunks:
        # 服务重启后内存中的 all_chunks 为空，从 Chroma 恢复历史 chunk
        all_chunks.extend(vectorstore.get()["documents"] or [])
    if not all_chunks:
        logger.warning("知识库为空，v3 检索器暂不构建（先通过 /rag/upload 上传文档）")
        return  # ensemble 保持 None

    bm25 = BM25Retriever.from_texts(all_chunks, k=5)          # 关键词（静态，全量重建）
    dense = vectorstore.as_retriever(search_kwargs={"k": 5})  # 语义（增量，无需重建）
    ensemble = EnsembleRetriever(retrievers=[dense, bm25], weights=[0.6, 0.4])
    final_retriever = ContextualCompressionRetriever(
        base_retriever=ensemble,
        base_compressor=compressor,
    )


refresh_retrievers()  # import 时尝试构建（空库时安全跳过，不会报错）

def _llm_call_with_retry(prompt: str, max_retries: int = 3) -> str:
    """带指数退避的 LLM 调用

    应对场景：API 限流(429)、服务不可用(503)、瞬时网络故障。
    退避策略：1s → 2s → 4s（指数增长，避免雪崩）
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            response = llm.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort=settings.llm_reasoning_effort,
                extra_body={
                    "thinking": {"type": settings.llm_thinking},
                },
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1, 2, 4 秒
                logger.warning("LLM 调用失败 (第 %d/%d 次)，%s 秒后重试: %s",
                               attempt + 1, max_retries, wait, e)
                time.sleep(wait)

    logger.error("LLM 调用重试 %d 次后仍失败: %s", max_retries, last_error)
    raise GenerationError(f"LLM 调用失败（已重试 {max_retries} 次）") from last_error


def _extract_text(file_path: str) -> str:
    """根据文件扩展名提取文本内容。支持 txt、pdf。"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        import pymupdf  # PyMuPDF——轻量、纯 Python，无需系统依赖
        doc = pymupdf.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    else:
        # 默认按 UTF-8 文本处理（.txt / .md / .csv 等）
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


def upload_document(file_path: str) -> int:
    """读取文档（txt/pdf）→ 切分 → 向量化 → 存入 Chroma

    Returns:
        chunk 数量（0 表示文件为空）
    """
    text = _extract_text(file_path)

    if not text.strip():
        return 0

    try:
        # ★ v4 新增：读取原始文件计算内容哈希，生成确定性 chunk ID
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:8]  # 取前 8 位，短且够用
        chunks = text_splitter.split_text(text)
        ids = [f"{file_hash}-{i}" for i in range(len(chunks))]  # 如 "a3f2c1e0-0"
        # ★ ids 只给 Chroma 内部索引用——search() 返回的 doc.metadata 不会自动包含它。
        #   必须同时传 metadatas，评估时才能从检索结果中拿到 chunk_id 做命中判定。
        vectorstore.add_texts(chunks, ids=ids, metadatas=[{"chunk_id": cid} for cid in ids])
        vectorstore.persist()
        all_chunks.extend(chunks)  # v3 BM25 用
        refresh_retrievers()  # v3：重建 BM25 索引，让新文档立即可被关键词检索
        logger.info("文档 %s 已处理：%d 个 chunk", file_path, len(chunks))
        return len(chunks)
    except Exception as e:
        logger.error("文档处理失败: %s", e)
        raise EmbeddingError(f"文档向量化失败: {e}") from e

REWRITE_SYSTEM_PROMPT = """把用户的模糊问题重写为适合检索的精确查询。
输出 2-3 个搜索查询，每行一个，不要输出其他内容。

示例：
用户问题：上次那个 bug 是怎么修的来着
重写为：
bug 修复方法 历史记录
缺陷修复 解决方案"""

def _should_rewrite(query: str) -> bool:
    """规则前置过滤——避免把精确 query 也送去 LLM 重写，浪费 API 调用和 ~1s 延迟。

    不是所有 query 都要重写——改写本身可能引入偏差。加一层轻量规则过滤，
    只放行"明显口语化/模糊"的 query。面试时这个判断逻辑本身就是加分项。
    """
    # 1. 太短的不需要（"你好"、"谢谢"）
    if len(query) < 8:
        return False
    # 2. 已含专业技术术语的跳过——本身就够精确
    tech_keywords = ["asyncio", "pydantic", "fastapi", "api", "docker", "sql"]
    if any(kw in query.lower() for kw in tech_keywords):
        return False
    # 3. 含时间限定词的跳过——改写可能丢掉时效信息
    time_keywords = ["最新", "2026", "今年", "当前版本", "最近"]
    if any(kw in query for kw in time_keywords):
        return False
    # 4. 含口语/疑问填充词 → 大概率需要重写
    colloquial = ["来着", "怎么搞", "啥", "咋", "有没有", "能不能", "怎么办", "什么意思"]
    if any(kw in query for kw in colloquial):
        return True
    # 5. 兜底：query 够长且无明显技术术语 → 可能模糊，送 LLM 试试
    return len(query) > 15

def rewrite_query(query: str) -> list[str]:
    """用 LLM 把模糊 query 重写为多个精确检索词

    输入："上次那个 bug 是怎么修的来着"
    输出：["bug 修复方法 历史记录", "缺陷修复 解决方案"]

    先经 _should_rewrite() 规则过滤——够精确的 query 直接跳过 LLM 调用。
    失败时回退到原始 query——不阻塞检索主流程。
    """
    if not _should_rewrite(query):
        logger.info("Query 已足够精确，跳过重写: %s", query)
        return [query]

    try:
        response = llm.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": f"用户问题：{query}\n重写为："},
            ],
            reasoning_effort=settings.llm_reasoning_effort,
            extra_body={
                "thinking": {"type": settings.llm_thinking},
            },
        )
        content = response.choices[0].message.content
        if not content:
            logger.warning("Query 重写返回空内容，回退到原始 query: %s", query)
            return [query]
        # 按行拆分，去掉空行和首尾空白
        queries = [q.strip() for q in content.strip().split("\n") if q.strip()]
        logger.info("Query 重写: %s → %s", query, queries)
        return queries if queries else [query]
    except Exception as e:
        logger.warning("Query 重写失败，回退到原始 query: %s, 错误: %s", query, e)
        return [query]


def search(query: str, k: int = 3) -> list[tuple[str, dict]]:
    """在向量库中检索最相关的 k 个文档片段

    Returns:
        [(page_content, metadata), ...]
    """
    try:
        docs = vectorstore.similarity_search(query, k=k)
        return [(doc.page_content, doc.metadata) for doc in docs]
    except Exception as e:
        logger.error("检索失败 (query=%s): %s", query[:50], e)
        raise RetrievalError(f"检索失败: {e}") from e

def search_v3(query: str) -> list[tuple[str, dict]]:
    """v3 完整检索：rewrite → ensemble(Dense+BM25) → Rerank → Top-3

    知识库为空（final_retriever 未构建）时自动回退 v1 纯向量检索——
    链路每一环都可降级，和 rewrite_query 失败回退是同一套思路。
    """
    if final_retriever is None:
        return search(query)

    queries = rewrite_query(query)  # 模糊 query → 2~3 个精确检索词（失败回退原 query）
    docs = []
    for q in queries:
        docs.extend(final_retriever.invoke(q))  # 每个检索词各自走 混合→Rerank

    # 多个重写词的结果可能重复：按内容精确去重，保序取前 3
    # ★ 升级路径：当前用精确匹配（set of str）。如果发现同一信息以不同措辞重复出现
    #   （如 "单文档上限 50 MB" ≈ "文件大小不超过 50 MB" ），可改为语义去重——
    #   直接 import v2 agent_tools 的 deduplicate_by_similarity(docs, threshold=0.85)
    #   替换下面这段精确去重逻辑。基建造好了，这里只是一行 import 的事。
    seen: set[str] = set()
    unique = []
    for d in docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            unique.append(d)
    return [(d.page_content, d.metadata) for d in unique[:3]]


def generate(query: str, context_docs: list | None = None) -> str:
    """拼接 Prompt → LLM 生成回答

    Args:
        query: 用户问题
        context_docs: search() 的返回值。传 None 则自动检索
    """
    if context_docs is None:
        context_docs = search_v3(query)

    context = "\n---\n".join([doc[0] for doc in context_docs])

    prompt = f"""根据以下参考资料回答问题。如果资料中没有相关信息，请如实说"未找到相关信息"。

参考资料：
{context}

问题：{query}
回答："""

    return _llm_call_with_retry(prompt)


# ============================================================
# 生产级改进方向（面试时可以说"当前是学习版，生产会做以下改进"）
# ============================================================
#
# 1. 懒加载工厂函数 —— 避免 import 时触发模型下载
#    def get_vectorstore() -> Chroma:
#        global _vectorstore
#        if _vectorstore is None:
#            _vectorstore = Chroma(...)
#        return _vectorstore
#    # v2 改为 from backend.services.rag_service import get_vectorstore
#
# 2. 依赖注入 —— 单元测试可以 mock
#    class RAGService:
#        def __init__(self, embedding, vectorstore, llm):
#            ...
#    然后用 FastAPI Depends() 注入
#
# 3. Async —— upload / search / generate 都是同步阻塞的
#    Chroma 0.6 暂不支持 async，生产可换 Milvus / Qdrant
#
# 4. 连接池 —— OpenAI 客户端复用 HTTP 连接池而非每次创建
#    （当前代码已经是复用的——llm 是模块级单例）
