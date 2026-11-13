"""对话模型构造：Agent 编排层与固定管线 baseline 共用的最底层依赖。

为什么单独成模块而不是留在 rag/chain.py：
    chain.py 是「固定管线 baseline」这一被对比的实现。若 get_llm 定义在那里，
    agent/ 就得反向 import 一个对比实现——层次是倒的，将来删掉 baseline 会先
    弄断 agent。两边都从本模块取，二者才是平级的。
"""
from langchain_deepseek import ChatDeepSeek

from config import settings

# 与历史行为保持一致：未显式传温度时用 0.2（settings.llm_temperature 的默认值）
DEFAULT_TEMPERATURE = 0.2


def get_llm(temperature: float = DEFAULT_TEMPERATURE):
    """构造 DeepSeek 对话模型（仅用于生成回答，不用于向量）。

    密钥缺失时由 settings.require_deepseek() 抛出可读错误，便于 --rebuild
    这类只需要 Embedding 的场景绕开对话模型独立运行。
    """
    return ChatDeepSeek(
        api_key=settings.require_deepseek(),
        model=settings.deepseek_model,
        temperature=temperature,
    )
