"""LangGraph 版的 ReAct 循环。

图结构：

    START → agent ──(模型请求调用工具)──→ tools ──→ agent ...
              │
              ├──(模型给出最终答案)──→ END
              └──(达到步数上限)────→ guard ──→ END

与 rag/chain.py 固定管线的核心差别：
"要不要检索、检索几次、要不要计算、要不要产出文件"由模型每轮自主决策，
而不是写死在代码路径里。固定管线保留为 baseline，用于效果与成本对比。
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.prompts import SYSTEM_PROMPT
from agent.state import AgentState
from agent.tools import make_tools
from config import settings
from rag.chain import get_llm

STEP_LIMIT_MESSAGE = (
    "我已连续推理 {n} 步仍未得到确定结论。为避免陷入无效循环，"
    "先给出目前掌握的信息；如需更精确的结果，请缩小问题范围后重试。"
)


def build_agent(db, checkpointer=None):
    """编译 Agent 图，返回 (可调用图, 引用收集列表)。

    checkpointer 决定多轮记忆如何保存：默认 MemorySaver 存在进程内存里，
    进程退出即丢失；后续换成 SQLite/Redis 版本即可持久化，图本身不用改。
    """
    collected: list[dict] = []
    tools = make_tools(db, collected)
    llm = get_llm(settings.llm_temperature).bind_tools(tools)
    max_steps = settings.agent_max_steps

    def agent_node(state: AgentState):
        messages = state["messages"]
        # 系统提示词每轮现注入、不写进历史，避免多轮后重复占用上下文
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
        response = llm.invoke(messages)
        return {
            "messages": [response],
            "steps": state.get("steps", 0) + 1,
        }

    def guard_node(state: AgentState):
        """步数超限的兜底：先补齐悬空的 tool_call，再给一条收尾消息。

        为什么不能直接跳 END：最后一条 AI 消息可能带着 tool_calls，
        若历史里留下未被响应的 tool_call，下一次请求会被 OpenAI 兼容接口直接拒绝。
        """
        patch: list = []
        for call in getattr(state["messages"][-1], "tool_calls", None) or []:
            patch.append(
                ToolMessage(
                    content="已达最大推理步数，本次工具调用被跳过。",
                    tool_call_id=call["id"],
                )
            )
        patch.append(AIMessage(content=STEP_LIMIT_MESSAGE.format(n=max_steps)))
        return {"messages": patch}

    def route(state: AgentState) -> str:
        if state.get("steps", 0) >= max_steps:
            return "guard"
        if getattr(state["messages"][-1], "tool_calls", None):
            return "tools"
        return "done"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("guard", guard_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", route, {"tools": "tools", "guard": "guard", "done": END}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("guard", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver()), collected


def ask(agent, collected: list[dict], question: str, thread_id: str = "default"):
    """执行一轮问答，返回 (最终答案, 本轮引用列表)。

    thread_id 相同即视为同一会话，历史由 checkpointer 自动接续；
    steps 每轮重置为 0，因此 max_steps 限制的是"单轮"推理步数而非累计。
    """
    collected.clear()
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)], "question": question, "steps": 0},
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": settings.agent_max_steps * 2 + 5,
        },
    )
    return result["messages"][-1].content, list(collected)
