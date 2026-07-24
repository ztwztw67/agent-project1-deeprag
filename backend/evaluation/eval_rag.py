import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ⚠️ 先加载 .env，再 import rag_service。
# rag_service 模块顶层会初始化 llm = OpenAI(api_key=...)，未加载 .env 时报 Missing credentials。
# 评估只测检索（无需 LLM），但模块顶层的 llm 初始化绕不过去——生产改进：把 llm 改为懒加载。
from dotenv import load_dotenv
load_dotenv()

from backend.services.rag_service import search, search_v3


# ====== Chunk ID 发现工具 ======
def show_chunks():
    """打印知识库中所有 chunk 的 ID 和内容预览。

    上传 test_doc.pdf 后运行一次，根据输出把 chunk_id 填入下方 test_queries 的
    relevant_chunk_ids 字段，替换 3b9cd67e 占位符。
    运行方式：python -c "from backend.evaluation.eval_rag import show_chunks; show_chunks()"
    """
    from backend.services.rag_service import vectorstore
    data = vectorstore.get()
    for i, (chunk_id, content) in enumerate(zip(data["ids"], data["documents"])):
        print(f"{chunk_id}  ←  {content[:80].replace(chr(10), ' ')}")


# ====== 测试集 ======
# 构造方法（参考，面试时讲这个流程比讲"我有 30 条数据"更有价值）：
#   ① 用户历史 query：从 /rag/chat 的实际日志中提取真实问题（最贴近生产）
#   ② 文档反向生成：把文档分段，让 LLM 为每段生成一个"用户可能问的问题"
#      （例：SLA 段落 → "CS-Pro 的可用性保证是多少？"）
#   ③ LLM 合成 + 人工校验：用 GPT-4 批量生成 50 条 → 人工筛掉不合理/重复的 → 保留 20-30 条
#
# 标注方法（两套方案，主方案优先）：
#   主方案 — chunk ID：上传后运行 show_chunks() 查出 chunk_id → 人工确认哪个 chunk
#            包含正确答案 → 填入 relevant_chunk_ids。评估时精确比对 chunk ID——
#            不受 PDF 换行截断、同义改写、关键词遗漏等干扰。
#   回退方案 — 关键词：当 chunk ID 未标注时（relevant_chunk_ids 为空），自动降级为
#              relevant_keywords 子串匹配（含空白归一）。

test_queries = [
    # === 基础召回（v1 应全对） ===
    {"query": "云枢科技的 CTO 是谁？",
     "relevant_chunk_ids": ["3b9cd67e-0"],       # ★ 上传后查 show_chunks() 替换
     "relevant_keywords": ["林远舟"]},
    {"query": "云枢科技什么时候成立的？",
     "relevant_chunk_ids": ["3b9cd67e-0"],
     "relevant_keywords": ["2019年3月"]},      # 关键词已做空白归一（原 "2019 年 3 月" 会被 PDF 换行截断）
    # === 表格/易混实体 ===
    {"query": "CS-Pro 一年多少钱？",
     "relevant_chunk_ids": ["3b9cd67e-1"],
     "relevant_keywords": ["128,000"]},
    {"query": "CS-Lite 的知识库存储是多大？",
     "relevant_chunk_ids": ["3b9cd67e-1"],
     "relevant_keywords": ["100GB"]},
    {"query": "CS-Pro 的 SLA 可用性承诺是多少？",
     "relevant_chunk_ids": ["3b9cd67e-2"],
     "relevant_keywords": ["99.95%"]},
    # === 否定/例外 ===
    {"query": "每月维护窗口的停机算不算 SLA 违约？",
     "relevant_chunk_ids": ["3b9cd67e-3"],
     "relevant_keywords": ["不计入SLA"]},
    {"query": "SLA 补偿可以退现金吗？",
     "relevant_chunk_ids": ["3b9cd67e-2"],
     "relevant_keywords": ["不以现金"]},
    # === 时效冲突 ===
    {"query": "单文档上传上限是多少？",
     "relevant_chunk_ids": ["3b9cd67e-4"],
     "relevant_keywords": ["50MB"]},           # PDF 换行截断成 "50\nMB"，归一后匹配
    # === 诚实性（文档无答案） ===
    {"query": "云枢科技的 CEO 是谁？",
     "relevant_chunk_ids": [],                  # 故意留空——文档无此信息
     "relevant_keywords": []},
    {"query": "智言平台怎么申请退款？",
     "relevant_chunk_ids": [],
     "relevant_keywords": []},
]
# 建议至少扩展到 20 条——太少没有统计意义，太多标注成本高


# ====== 评估函数 ======
def _norm(s: str) -> str:
    """空白归一：PDF 换行把 '50\nMB' 拆成两行，归一后才能匹配关键词 '50MB'。"""
    return re.sub(r"\s+", "", s)


def _get_retrieved_ids(results) -> set[str]:
    """从检索结果中提取 chunk_id 集合。"""
    ids = set()
    for r in results:
        if isinstance(r, tuple):
            cid = (r[1] or {}).get("chunk_id", "")
        elif hasattr(r, "metadata"):
            cid = (r.metadata or {}).get("chunk_id", "")
        else:
            cid = ""
        if cid:
            ids.add(cid)
    return ids


def _get_retrieved_texts(results) -> list[str]:
    """从检索结果中提取文本列表。"""
    return [r[0] if isinstance(r, tuple) else r.page_content for r in results]


def evaluate_hit_rate(test_set, retriever_fn) -> float:
    """Hit Rate：正确事实出现在 Top-3 结果中的 query 占比。

    判定优先级：chunk ID 精确匹配 > 关键词子串匹配（含空白归一）。
    """
    hits = 0
    for item in test_set:
        results = retriever_fn(item["query"])
        target_ids = item.get("relevant_chunk_ids", [])
        keywords = item.get("relevant_keywords", [])

        # 主方案：chunk ID 精确匹配
        if target_ids:
            retrieved_ids = _get_retrieved_ids(results)
            if any(tid in retrieved_ids for tid in target_ids):
                hits += 1
                continue  # 命中，跳过回退方案
            # chunk ID 未命中——说明检索确实没召回正确 chunk
            texts = _get_retrieved_texts(results)
            print(f"  ✗ chunk ID 未命中: {item['query'][:40]}  |  "
                  f"target={target_ids}  retrieved={retrieved_ids}  "
                  f"top1 前60字: {_norm(texts[0])[:60] if texts else '-'}")
            continue

        # 回退方案：关键词子串匹配（chunk ID 未标注时启用）
        if not keywords:
            continue  # 诚实性测试，不计入分母
        texts = _get_retrieved_texts(results)
        combined = _norm(" ".join(texts))
        if any(_norm(kw) in combined for kw in keywords):
            hits += 1
    total = sum(1 for t in test_queries
                if t.get("relevant_chunk_ids") or t.get("relevant_keywords"))
    return hits / total if total > 0 else 0.0


def evaluate_mrr(test_set, retriever_fn) -> float:
    """MRR（Mean Reciprocal Rank）：正确答案首次出现排名的倒数平均值。

    排名从 1 开始（排第 1 位 = 1/1 = 1.0；排第 10 位 = 1/10 = 0.1）。
    只取第一个匹配的结果的排名。判定优先级同 evaluate_hit_rate。
    """
    reciprocal_ranks = []
    for item in test_set:
        results = retriever_fn(item["query"])
        target_ids = item.get("relevant_chunk_ids", [])
        keywords = item.get("relevant_keywords", [])

        # 主方案：chunk ID
        if target_ids:
            for rank, r in enumerate(results, start=1):
                cid = ""
                if isinstance(r, tuple):
                    cid = (r[1] or {}).get("chunk_id", "")
                elif hasattr(r, "metadata"):
                    cid = (r.metadata or {}).get("chunk_id", "")
                if cid and cid in target_ids:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)
            continue

        # 回退方案：关键词
        if not keywords:
            continue
        texts = _get_retrieved_texts(results)
        for rank, text in enumerate(texts, start=1):
            if any(_norm(kw) in _norm(text) for kw in keywords):
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


# ====== 运行评估 ======
if __name__ == "__main__":
    # show_chunks()
    print("=" * 60)
    print("RAG 评估 —— v1 纯向量 vs v3 混合+Rerank")
    print(f"测试集大小: {len(test_queries)} 条")
    print("=" * 60)

    for label, fn in [("v1 纯向量 search()", search), ("v3 混合+Rerank search_v3()", search_v3)]:
        hr = evaluate_hit_rate(test_queries, fn)
        mrr = evaluate_mrr(test_queries, fn)
        print(f"\n{label}:")
        print(f"  Hit Rate = {hr:.2%}")
        print(f"  MRR      = {mrr:.3f}")

    print("\n💡 面试话术：'我对比了 v1 纯向量和 v3 混合+Rerank 两个版本。"
          "v1 的 Hit Rate 是 X%，v3 提升到 Y%，且 MRR 从 A 提升到 B——"
          "说明混合检索不仅让更多正确答案被找到，还让它们排得更靠前。'")