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
from rag.loader import load_pdf, split_docs
from rag.vector_store import (
    build_vector_db, load_vector_db, save_vector_db,
    add_to_vector_db, delete_from_vector_db, DEFAULT_INDEX_PATH,
)
from rag.manifest import file_hash, load_manifest, save_manifest

DATA_DIR = str(settings.data_dir)


def scan_pdf_files():
    os.makedirs(DATA_DIR, exist_ok=True)     # 首次运行时 data 目录可能还不存在
    result = {}
    for f in os.listdir(DATA_DIR):          # 列出 data 目录下所有文件名
        if f.lower().endswith(".pdf"):      # 只要 .pdf 结尾的（忽略大小写）
            path = os.path.join(DATA_DIR, f)             # 拼出完整路径
            h = file_hash(path)                          # 算这个文件内容的 sha256
            result[h] = path                             # 哈希做键、路径做值
    return result


def index_documents(force_rebuild: bool = False):
    """增量维护索引：只对新 PDF / 内容变化的 PDF 做切块和向量化。"""
    manifest = {} if force_rebuild else load_manifest()
    current = scan_pdf_files()
    if not current:
        raise FileNotFoundError(f"{DATA_DIR} 下没有找到任何 PDF 文件")

    index_exists = os.path.exists(os.path.join(DEFAULT_INDEX_PATH, "index.faiss"))

    # 需要删除的：登记过但文件已被移除（内容变化的文件哈希会变，等同于旧哈希被删除）
    to_delete = {h: info for h, info in manifest.items() if h not in current}
    # 需要新增的：没登记过的文件（新文件或修改后的文件）
    to_add = [p for h, p in current.items() if h not in manifest]

    if not to_delete and not to_add and index_exists:
        print("索引已是最新，直接加载。")
        return load_vector_db()

    # 全量重建时不加载旧库，从头构建；否则有旧库就加载以便追加/删除
    db = load_vector_db() if (index_exists and not force_rebuild) else None

    # 1. 删除已失效文件的向量（稳定 ID 保证可精准删除）
    for h, info in to_delete.items():
        if db is not None and info.get("ids"):
            delete_from_vector_db(db, info["ids"])
        print(f"已移除旧向量: {info['source']}")
        manifest.pop(h, None)

    # 2. 只对新增/变更文件切块 + 向量化
    for path in to_add:
        name = os.path.basename(path)
        h = file_hash(path)
        print(f"正在索引新文件: {name}")
        chunks = split_docs(load_pdf(path))
        # 稳定 ID：文件哈希 + 块序号，同一内容永远同一 ID，便于后续删除
        ids = [f"{h}_{i}" for i in range(len(chunks))]
        if db is None:
            db = build_vector_db(chunks, ids=ids)
        else:
            add_to_vector_db(db, chunks, ids)
        manifest[h] = {"source": name, "ids": ids}
        print(f"  -> 新增 {len(chunks)} 块（其他文件向量未重算）")

    # 3. 持久化索引与登记簿
    save_vector_db(db)
    save_manifest(manifest)
    print(f"更新完成，当前共登记 {len(manifest)} 个文件。")
    return db


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