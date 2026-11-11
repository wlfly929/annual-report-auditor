import asyncio
import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import settings

# 基于文件位置定位项目根目录，读取 .env
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# RAG 提示词：要求模型仅根据检索到的上下文回答，减少幻觉
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个严谨的知识问答助手。请只根据下面提供的【参考资料】回答问题。"
     "如果参考资料里没有足够信息，请直接说明“文档中未找到相关信息”，不要编造。"),
    ("user",
     "【参考资料】\n{context}\n\n【问题】\n{question}"),
])


def get_llm(temperature: float = 0.2):
    """构造 DeepSeek 对话模型（仅用于生成回答，不用于向量）。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未检测到 DEEPSEEK_API_KEY，请检查项目根目录 .env 是否已保存"
        )
    return ChatDeepSeek(
        api_key=api_key,
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=temperature,
    )


def _format_docs(docs):
    """把检索到的文档片段拼成带编号上下文的字符串。"""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        blocks.append(f"[资料{i}] {doc.page_content.strip()}")
    return "\n\n".join(blocks)


def build_rag_chain(db, k: int | None = None):
    """基于 FAISS 向量库构建 RAG 链：检索 -> 组装提示词 -> LLM -> 文本输出。

    k 默认取 settings.retrieval_top_k，不能写死。历史上 chain 里写死 k=4、
    config 后来改成 8，而 main.py 调用时没传 k，于是 CLI 的 --baseline 和
    评测里的 baseline 跑在两套参数上，同一份“baseline”指标对不上。
    """
    k = settings.retrieval_top_k if k is None else k
    retriever = db.as_retriever(search_kwargs={"k": k})
    llm = get_llm()
    parser = StrOutputParser()

    def chain_fn(question: str):
        docs = retriever.invoke(question)
        context = _format_docs(docs)
        result = (RAG_PROMPT | llm | parser).invoke(
            {"context": context, "question": question}
        )
        return result, docs

    return chain_fn


def ask(question: str, db, k: int | None = None):
    """便捷函数：给定问题和向量库，返回 (回答, 命中文档)。"""
    chain_fn = build_rag_chain(db, k=k)
    return chain_fn(question)


async def astream_answer(question: str, db, k: int | None = None):
    """固定管线的流式版，产出 ("sources", docs) 后逐 token 产出 ("token", text)。

    先检索再生成，所以把命中文档作为第一个事件交出去，前端可以立刻展示引用，
    不必等答案生成完。检索含一次 embedding 网络调用，丢到线程里避免阻塞事件循环。
    """
    k = settings.retrieval_top_k if k is None else k
    docs = await asyncio.to_thread(
        lambda: db.as_retriever(search_kwargs={"k": k}).invoke(question)
    )
    yield "sources", docs

    chain = RAG_PROMPT | get_llm() | StrOutputParser()
    async for piece in chain.astream({"context": _format_docs(docs), "question": question}):
        if piece:
            yield "token", piece
