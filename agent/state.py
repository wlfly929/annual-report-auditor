"""Agent 共享状态。

LangGraph 的每个节点都读写同一份状态，字段增减会直接影响图的连通性，
所以单独抽到本文件，避免 tools / graph 之间循环导入。
"""
from typing import Annotated, TypedDict

from langgraph.graph import add_messages


class AgentState(TypedDict, total=False):
    # 完整对话历史。Annotated + add_messages 表示节点返回的消息是"追加"而非覆盖，
    # 这是多轮记忆能生效的前提。
    messages: Annotated[list, add_messages]

    # 本轮用户问题（脱离 messages 单独存一份，方便日志与后续评测）
    question: str

    # 已执行的模型决策轮次，用于 max_steps 守卫，防止死循环
    steps: int

    # 累计命中的资料 [{file, page, snippet}]，由工具闭包收集后写入
    sources: list[dict]
