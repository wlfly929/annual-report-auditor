"""端到端轨迹追踪：打印 Agent 每一步的决策与工具返回。

用法：
    python scripts/trace_agent.py "你的问题"     # 真实调用，查看 ReAct 轨迹
    python scripts/trace_agent.py --guard       # 用假模型验证步数守卫（不消耗 API）

--guard 模式刻意构造一个"永远请求调用工具"的假模型，用来稳定复现
「步数用尽 -> guard 收尾」这条分支；真实模型几乎不会自然走到这里。
"""
import sys
from pathlib import Path

# 允许 `python scripts/trace_agent.py` 直接运行（此时 sys.path[0] 是 scripts/）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

import agent.graph as graph_mod  # noqa: E402
from agent.graph import build_agent  # noqa: E402
from config import settings  # noqa: E402
from rag.indexer import index_documents  # noqa: E402

DEFAULT_QUESTION = (
    "请梳理这份简历里的项目经历和技术技能，"
    "并判断作者更适合前端开发还是 Agent 应用开发岗位，给出依据。"
)


def trace(agent, question, thread_id="trace-session"):
    """以 stream 模式跑一轮，逐节点打印轨迹。"""
    print("=" * 72)
    print(f"问题：{question}")
    print("=" * 72)

    payload = {"messages": [HumanMessage(content=question)], "question": question, "steps": 0}
    cfg = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.agent_max_steps * 2 + 5,
    }

    tick, answer = 0, "(未产出答案)"
    for chunk in agent.stream(payload, config=cfg, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "agent":
                tick += 1
                msg = update["messages"][-1]
                calls = getattr(msg, "tool_calls", None) or []
                if calls:
                    print(f"\n[步骤 {tick}] agent 决策：请求调用工具")
                    for call in calls:
                        print(f"    -> {call['name']}({str(call['args'])[:160]})")
                else:
                    print(f"\n[步骤 {tick}] agent 决策：信息已足够，输出最终答案")
                    answer = msg.content
            elif node == "tools":
                for msg in update["messages"]:
                    name = getattr(msg, "name", "tool")
                    body = str(msg.content).replace("\n", " ")
                    suffix = " ..." if len(body) > 220 else ""
                    print(f"    <- {name} 返回：{body[:220]}{suffix}")
            elif node == "guard":
                print(f"\n[守卫] 达到 AGENT_MAX_STEPS={settings.agent_max_steps}，强制收敛")
                for msg in update["messages"]:
                    if isinstance(msg, AIMessage):
                        answer = msg.content

    print("\n" + "-" * 72)
    print("最终答案：")
    print(answer)
    print("-" * 72)
    return answer


def run_real(question):
    db = index_documents()
    agent, collected = build_agent(db)
    trace(agent, question)

    print(f"\n本次会话累计命中片段 {len(collected)} 条，去重后来源：")
    seen = set()
    for item in collected:
        key = (item["file"], item["page"])
        if key not in seen:
            seen.add(key)
            print(f"  - {item['file']} p.{item['page']}")


class AlwaysToolCallModel:
    """永远请求调用 calculate 的假模型。"""

    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        return AIMessage(
            content="",
            tool_calls=[{
                "name": "calculate",
                "args": {"expression": f"{self.calls}+1"},
                "id": f"call_{self.calls}",
            }],
        )


def run_guard_test():
    """把重试上限压到 3 步，验证超限后 guard 能正确收尾。"""
    original_get_llm = graph_mod.get_llm
    original_max = settings.agent_max_steps

    graph_mod.get_llm = lambda *a, **k: AlwaysToolCallModel()
    settings.agent_max_steps = 3
    try:
        agent, _ = build_agent(None)
        trace(agent, "验证步数守卫（假模型，不调用真实 API）")
    finally:
        graph_mod.get_llm = original_get_llm
        settings.agent_max_steps = original_max


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if "--guard" in sys.argv:
        run_guard_test()
    else:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        run_real(" ".join(args) if args else DEFAULT_QUESTION)
