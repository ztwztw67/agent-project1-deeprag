from langchain_core.tools import tool                       # ✅ 0.3.x 正确路径
from langgraph.prebuilt import create_react_agent           # ✅ 0.3.x 推荐用法
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent  # ✅ 0.3.x 正确路径（包级导出）
from langchain_core.documents import Document               # v3 normalize_to_documents 用
from langchain_openai import ChatOpenAI                     # ✅ 兼容所有 OpenAI 协议 API（DeepSeek 等）

# ⚠️ 跨文件依赖——vectorstore、embedding_model 在 v1 的 rag_service.py 中初始化
from backend.services.rag_service import vectorstore, embedding_model
from backend.config import settings

# ====== LangChain Chat Model（Agent 专用） ======
# ⚠️ create_react_agent / create_sql_agent 需要 .bind_tools() 方法，
# 原生 openai.OpenAI 没有这个方法，所以这里用 LangChain 的 ChatOpenAI 封装
# ChatOpenAI 走 OpenAI 协议，传 base_url 后实际请求的是 DeepSeek（或任何兼容接口）
chat_llm = ChatOpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url,
    model=settings.llm_model,
)

# ====== Tool 1: 本地文档检索 ======
@tool
def search_documents(query: str) -> str:
    """在公司内部知识库中搜索。适用于：政策查询、技术文档、历史记录。"""
    from backend.services.rag_service import search_v3  # ★ 改为 import search_v3
    results = search_v3(query)  # ★ 混合检索+Rerank；final_retriever 未构建时自动回退 search()
    return "\n---\n".join([doc[0] for doc in results])

# ====== Tool 2: Web 搜索 ======
web_search = TavilySearchResults(
    max_results=3,
    search_depth="advanced",  # "advanced"=精准但慢（~3s），"basic"=快速但略糙（~1s）
    tavily_api_key=settings.tavily_api_key,  # 从 .env 的 TAVILY_API_KEY 读取
)                            # Tavily 返回的结果已经是结构化的：title + url + content                        # Tavily 返回的结果已经是结构化的：title + url + content

# ====== Tool 3: SQL 查询 ======
_sql_agent = None  # 延迟初始化，避免 import 时就连接数据库

def _get_sql_agent():
    """延迟初始化 SQL Agent——只在第一次实际调用时连接数据库。
    放在函数里而非模块级别，好处：
    ① import agent_tools 时不会因数据库不可用而崩溃
    ② 不用 SQL 工具时完全不触发连接
    """
    global _sql_agent
    if _sql_agent is None:
        # ⚠️ 替换为实际数据库连接串
        # 驱动选 pymysql（纯 Python，pip install pymysql）而非 mysqldb（需编译 mysqlclient）
        db = SQLDatabase.from_uri("mysql+pymysql://user:password@localhost:3306/mydb")
        _sql_agent = create_sql_agent(chat_llm, db=db, verbose=False)
    return _sql_agent

def normalize_to_documents(results: list, source: str) -> list[Document]:
    """将不同来源的检索结果统一为 LangChain Document 格式

    Args:
        results: 原始检索结果，格式因 source 而异
                 - "local":   [(page_content, metadata), ...]  ← rag_service.search() 返回
                 - "web":     [{"content": ..., "url": ...}, ...]  ← Tavily 返回
                 - "sql":     str（自然语言回答）                   ← text-to-SQL 返回
        source:   "local" | "web" | "sql"
    """
    if source == "local":
        return [
            Document(page_content=content, metadata={**meta, "source": "local"})
            for content, meta in results
        ]
    elif source == "web":
        return [
            Document(
                page_content=r.get("content", ""),
                metadata={"source": "web", "url": r.get("url", "")},
            )
            for r in results
        ]
    elif source == "sql":
        # SQL 返回的是自然语言摘要，直接包装成一个 Document
        return [Document(page_content=str(results), metadata={"source": "sql"})]
    return []


def deduplicate_by_similarity(
    docs: list[Document], threshold: float = 0.85
) -> list[Document]:
    """基于文本相似度去重——两条内容相似度>threshold 视为重复，保留第一条

    为什么不用精确匹配？同一信息在不同源中表述不同（"北京人口2100万" vs
    "北京市常住人口约为2100万人"），精确匹配 == 去不掉，需要用语义相似度。

    threshold 经验值：0.85（过高去不掉，过低会误删。0.8~0.9 是常用区间）
    """
    if len(docs) <= 1:
        return docs

    from sentence_transformers import util

    # ✅ 复用 v1 的 embedding_model.client，避免重复加载 ~100MB 模型
    contents = [d.page_content for d in docs]
    embeddings = embedding_model.client.encode(contents, convert_to_tensor=True)

    kept = []
    for i, doc in enumerate(docs):
        is_dup = False
        for j in kept:
            sim = util.cos_sim(embeddings[i], embeddings[j]).item()
            if sim > threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(i)
    return [docs[i] for i in kept]

@tool
def query_database(question: str) -> str:
    """在业务数据库中查询。适用于：统计数据、用户信息、订单记录。
    请输入自然语言问题，我会转换为 SQL。"""
    # create_sql_agent 内部做的事：
    #   ① 读取数据库 schema（表名、字段名）
    #   ② 把 schema + 用户问题给 LLM → 生成 SQL
    #   ③ 执行 SQL → 把结果 + 问题给 LLM → 生成自然语言回答
    result = _get_sql_agent().invoke({"input": question})
    return result.get("output", str(result))  # ✅ 用 .get() 防 KeyError

# ====== 组合：Agent 自主选择 ======
# 三个 tool 放进同一个列表 → Agent 收到 Query 后自己判断用哪个
tools = [search_documents, web_search, query_database]

# langgraph 的 create_react_agent 不需要手写 prompt —— 自动将 tool 的 name+description 注入
agent = create_react_agent(chat_llm, tools)
# invoke 时传 messages 格式（langgraph 标准接口）：
# result = agent.invoke({"messages": [("human", query)]})
# answer = result["messages"][-1].content  # 取最后一条 AI 消息


