"""测试脚本：验证 LangChain Agent 能正常跑通"""
from dotenv import load_dotenv
load_dotenv()  # ← 必须放在最前面，让后续 import 能读到环境变量

from langchain_openai import ChatOpenAI
from langchain.agents import tool , create_react_agent , AgentExecutor
from langchain.prompts import ChatPromptTemplate

# Step 1: 定义工具——这是 Agent 的核心，工具决定了 Agent 能做什么
@tool
def search_knowledge_base(query: str) -> str:
    """在内部知识库中搜索。当用户问「...是什么」「查一下...」时，优先用这个工具。"""
    return f"知识库中关于'{query}'的结果：..."

@tool
def calculate(expression: str) -> str:
    """执行数学计算。当用户要求「计算」「算一下」「等于多少」时，用这个工具。
    输入格式示例：'123 + 456' 或 '3 * 7'（只支持加减乘除和括号）"""
    import re
    # 安全过滤：只允许数字、运算符、空格、括号、小数点
    if not re.match(r'^[\d\s+\-*/().]+$', expression):
        return "错误：表达式包含不允许的字符"
    try:
        # 用 eval 但前面已经做了安全过滤，只允许数学表达式
        return str(eval(expression))
    except Exception as e:
        return f"计算失败：{e}"

@ tool
def search_weather(location: str) -> str:
    """查询天气。当用户问「今天天气如何」[查询实时天气]时，用这个工具。"""
    import requests
    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid=YOUR_API_KEY"
    response = requests.get(url)
    data = response.json()
    if response.status_code == 200:
        return f"{data['name']} 的天气是 {data['weather'][0]['description']}，温度是 {data['main']['temp']} 度。"
    else:
        return "无法获取天气信息"

# Step 2: 初始化 LLM
# reasoning_effort 是 ChatOpenAI 原生参数，langchain-openai 会正确映射到 API
# thinking 是 DeepSeek 特有字段，需要通过 extra_body 直接写入请求体
llm = ChatOpenAI(
    model="deepseek-v4-pro",
    temperature=0,
    reasoning_effort="high",
    extra_body={
        "thinking": {"type": "enabled"},
    }
)

# Step 3: 创建 Agent
tools = [search_knowledge_base, calculate]
# ⚠️ ReAct prompt 的关键：必须用英文标记 Thought/Action/Action Input/Final Answer
# 这是解析器硬编码的格式，不能翻译成中文
#
# agent_scratchpad 是中间步骤（Thought+Action+Observation）的累积记录，
# 每次循环都会被追加，模型看到历史就知道自己做过什么
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个能使用工具的助手。用中文回答用户。

    可用工具：
    {tools}
    
    工具名称：{tool_names}
    
    严格按以下格式输出（标记必须用英文）：
    Thought: [你接下来应该做什么，中文]
    Action: [工具名称]
    Action Input: [输入参数，单行]
    Observation: [工具返回的结果——这是系统自动填入的，你不用写]
    ... (可重复多轮)
    Thought: 我现在知道答案了
    Final Answer: [最终回答，中文]
    
    注意：
    - 每次只调一个工具
    - 得到 Observation 后，如果信息够了就直接 Final Answer，不要重复调同一个工具
    - 如果不需要工具，直接输出 Final Answer"""),
    ("human", "{input}\n\n{agent_scratchpad}"),
])

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,
)

# Step 4: 运行
if __name__ == "__main__":
    print("=" * 50)
    print("测试 1：需要搜索的问题")
    print("=" * 50)
    result = executor.invoke({"input": "什么是RAG？"})
    print(f"\n最终回答：{result['output']}\n")

    print("=" * 50)
    print("测试 2：需要计算的问题")
    print("=" * 50)
    result = executor.invoke({"input": "计算 123 + 456 等于多少"})
    print(f"\n最终回答：{result['output']}\n")

    print("=" * 50)
    print("测试 3：先搜后算（两个工具连用）")
    print("=" * 50)
    result = executor.invoke({"input": "查一下湖南最近的天气"})
    print(f"\n最终回答：{result['output']}")
