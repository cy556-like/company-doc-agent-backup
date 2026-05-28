"""
Agent 核心逻辑模块
使用 LangGraph 构建 ReAct 模式的 Agent
ReAct = Reasoning(推理) + Acting(行动) → 边思考边行动
优化：限制历史消息数量 + 限制最大工具调用轮数 + 动态日期注入 + 知识库优先
"""
from datetime import datetime
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from app.config import settings
from app.agent.tools import ALL_TOOLS
from app.agent.prompts import SYSTEM_PROMPT
from app.memory.manager import get_session_history

# 最大历史消息数量（加速推理，避免上下文过长）
MAX_HISTORY_MESSAGES = 10

# 最大工具调用轮数（防止无限循环）
MAX_TOOL_ROUNDS = 3


def _build_system_prompt() -> str:
    """构建动态系统提示词，注入当前日期和知识库优先规则"""
    now = datetime.now()
    current_date = now.strftime("%Y年%m月%d日")
    current_weekday = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][now.weekday()]

    date_instruction = f"""

## 重要：当前时间信息
当前日期：{current_date} {current_weekday}
你在回答问题时，必须使用上述真实日期，不得编造或猜测任何日期信息。
当用户询问与时间相关的问题时，请以当前日期为基准进行计算和回答。
引用文档内容时，如果文档中没有明确日期，不要自行添加或推测日期。

## 知识库优先规则（必须严格遵守）
1. 当用户询问任何与公司制度、流程、规范、人员、文档相关的问题时，你必须首先调用 search_documents_tool 搜索知识库
2. 只有在知识库中找不到相关内容时，才使用自身知识回答，并明确告知用户"知识库中未找到相关内容，以下为AI参考回答"
3. 绝对不要在知识库有相关内容的情况下跳过检索直接用AI知识回答
4. 回答必须引用知识库中的实际文档内容，不得编造"""

    return SYSTEM_PROMPT + date_instruction


# ===== 1. 定义 Agent 状态 =====
class AgentState(TypedDict):
    """
    Agent 的状态定义
    messages 使用 add_messages 策略：新消息追加而非覆盖
    """
    messages: Annotated[list, add_messages]


# ===== 2. 创建 LLM =====
def create_llm():
    """创建 LLM 实例"""
    return ChatOpenAI(
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        model=settings.LLM_MODEL,
        temperature=0.1,  # 低温度 = 更确定性的回答
    )


# ===== 3. 构建 Agent 图 =====
def create_agent_graph():
    """
    构建 LangGraph Agent 执行图

    流程：用户输入 → LLM 思考 → 是否调用工具？
           ├─ 是 → 执行工具 → 回到 LLM 思考（循环，最多3轮）
           └─ 否 → 输出回答 → 结束
    """
    llm = create_llm()

    # 将工具绑定到 LLM，让它知道有哪些工具可以用
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # 节点1: LLM 思考节点
    def think(state: AgentState):
        """LLM 思考：分析用户问题，决定是否调用工具"""
        messages = state["messages"]
        # 动态构建系统提示词（注入当前日期+知识库优先规则）
        system_msg = SystemMessage(content=_build_system_prompt())
        response = llm_with_tools.invoke([system_msg] + messages)
        return {"messages": [response]}

    # 节点2: 工具执行节点（使用 LangGraph 内置的 ToolNode）
    tool_node = ToolNode(ALL_TOOLS)

    # 条件边：判断是否需要继续调用工具（限制最大轮数）
    def should_continue(state: AgentState):
        """
        判断是否需要继续调用工具
        如果工具调用轮数超过 MAX_TOOL_ROUNDS，则强制结束
        """
        messages = state["messages"]
        # 统计 ToolMessage 的数量，每轮工具调用会产生一个 ToolMessage
        tool_message_count = sum(1 for m in messages if isinstance(m, ToolMessage))

        if tool_message_count >= MAX_TOOL_ROUNDS:
            return END

        # 使用 LangGraph 内置判断：最后一条消息是否有工具调用
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "act"

        return END

    # ===== 构建状态图 =====
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("think", think)       # 思考节点
    graph.add_node("act", tool_node)     # 行动节点

    # 设置入口
    graph.set_entry_point("think")

    # 添加条件边：思考后判断是否需要调用工具
    graph.add_conditional_edges(
        "think",
        should_continue,
        {
            "act": "act",   # 需要工具 → 去执行
            END: END,       # 不需要工具 → 结束
        },
    )

    # 执行完工具后，回到思考节点（形成 ReAct 循环）
    graph.add_edge("act", "think")

    return graph.compile()


# ===== 4. 全局 Agent 实例 =====
_agent_graph = None


def get_agent():
    """获取 Agent 单例（懒加载）"""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = create_agent_graph()
    return _agent_graph


def chat(user_input: str, session_id: str = "default", agent_mode: bool = True, web_search: bool = True) -> str:
    """
    与 Agent 对话的核心方法

    Args:
        user_input: 用户输入
        session_id: 会话 ID（支持多用户）
        agent_mode: 是否使用智能体模式（默认True，只用Agent）
        web_search: 是否开启联网搜索（默认True）

    Returns:
        str: Agent 的回答
    """
    # 如果不是智能体模式，走纯LLM对话（目前默认始终走Agent模式）
    agent = get_agent()

    # 获取该会话的历史消息
    history = get_session_history(session_id)

    # 只取最近 MAX_HISTORY_MESSAGES 条历史消息（加速推理）
    recent_messages = history.messages[-MAX_HISTORY_MESSAGES:]

    # 如果联网搜索开启，在用户输入中附加提示
    enhanced_input = user_input
    if web_search:
        enhanced_input = f"[联网搜索已开启] {user_input}"

    # 构建完整的消息列表 = 历史消息 + 新消息
    all_messages = recent_messages + [HumanMessage(content=enhanced_input)]

    # 调用 Agent
    result = agent.invoke({"messages": all_messages})

    # 提取最后的 AI 回答
    ai_message = result["messages"][-1]

    # 保存到会话历史（保存原始用户输入，不含联网标记）
    history.add_message(HumanMessage(content=user_input))
    history.add_message(ai_message)

    return ai_message.content
