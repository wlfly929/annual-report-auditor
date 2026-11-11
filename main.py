"""文档研究 Agent 主入口。

用法：
    python main.py                      # 交互式问答（Agent 模式，多轮记忆）
    python main.py "你的问题"            # 单次提问，输出答案后退出
    python main.py --rebuild            # 强制全量重建索引
    python main.py --baseline "问题"     # 走原始固定管线 RAG，用于效果对比
"""
import os
import sys

from agent.graph import ask as ask_agent, build_agent
from config import settings
from rag.chain import ask as ask_baseline
from rag.indexer import index_documents


def print_sources(sources):
    """打印本轮引用来源。Agent 可能多次命中同一片段，这里按 (文件, 页码) 去重。"""
    unique, seen = [], set()
    for item in sources:
        key = (item["file"], item["page"])
        if key not in seen:
            seen.add(key)
            unique.append(item)

    if not unique:
        print("\n—— 本轮未引用文档 ——")
        return
    print("\n—— 参考资料 ——")
    for i, item in enumerate(unique, start=1):
        snippet = item["snippet"].replace("\n", " ")[:80]
        print(f"[{i}] {item['file']} (第{item['page']}页): {snippet}...")


def docs_to_sources(docs):
    """把 LangChain Document 转成与 Agent 一致的来源结构，让两种模式共用打印逻辑。"""
    sources = []
    for doc in docs:
        page = doc.metadata.get("page")
        sources.append({
            "file": os.path.basename(doc.metadata.get("source", "未知来源")),
            "page": page + 1 if isinstance(page, int) else "?",
            "snippet": doc.page_content.strip(),
        })
    return sources


def answer_question(question, db, agent, collected, use_baseline):
    """统一问答出口：--baseline 走固定管线，默认走 Agent。"""
    if use_baseline:
        answer, docs = ask_baseline(question, db)
        return answer, docs_to_sources(docs)
    return ask_agent(agent, collected, question, thread_id="cli-session")


def interactive_loop(db, agent, collected, use_baseline):
    print("\n进入问答模式，输入问题回车即可，输入 quit / exit / q 退出。")
    while True:
        try:
            question = input("\n你问> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("再见！")
            break
        answer, sources = answer_question(question, db, agent, collected, use_baseline)
        print(f"\nAI> {answer}")
        print_sources(sources)


def main():
    rebuild = "--rebuild" in sys.argv
    use_baseline = "--baseline" in sys.argv
    # 收集非参数作为一次性问题
    question_args = [a for a in sys.argv[1:] if not a.startswith("--")]

    db = index_documents(force_rebuild=rebuild)

    agent, collected = None, None
    if use_baseline:
        print("运行模式：baseline（原始固定管线 RAG）")
    else:
        agent, collected = build_agent(db)
        print(f"运行模式：Agent（单轮最多 {settings.agent_max_steps} 步推理）")

    if question_args:
        question = " ".join(question_args)
        answer, sources = answer_question(question, db, agent, collected, use_baseline)
        print(f"问题：{question}")
        print(f"回答：{answer}")
        print_sources(sources)
    else:
        interactive_loop(db, agent, collected, use_baseline)


if __name__ == "__main__":
    main()