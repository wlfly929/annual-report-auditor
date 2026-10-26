"""RAG 项目主入口。

用法：
    python main.py                 # 交互式问答（自动建库/加载索引）
    python main.py --rebuild       # 强制用 data 目录下的 PDF 重建索引
    python main.py "你的问题"       # 单次提问，直接输出答案后退出
"""
import os
import sys

from rag.loader import load_pdf, split_docs
from rag.vector_store import build_vector_db, load_vector_db, DEFAULT_INDEX_PATH
from rag.chain import ask

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def build_index_from_data():
    """读取 data 目录下所有 PDF，切分并构建 FAISS 索引。"""
    pdf_files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        raise FileNotFoundError(f"{DATA_DIR} 下没有找到任何 PDF 文件")

    all_chunks = []
    for path in pdf_files:
        print(f"正在解析: {os.path.basename(path)}")
        docs = load_pdf(path)
        chunks = split_docs(docs)
        print(f"  -> 切分 {len(chunks)} 块")
        all_chunks.extend(chunks)

    print(f"共 {len(all_chunks)} 个文本块，正在写入向量索引...")
    db = build_vector_db(all_chunks)
    print(f"索引已保存到: {DEFAULT_INDEX_PATH}")
    return db


def get_db(rebuild: bool = False):
    """存在索引则加载，否则构建；rebuild=True 时强制重建。"""
    index_ready = os.path.exists(os.path.join(DEFAULT_INDEX_PATH, "index.faiss"))
    if index_ready and not rebuild:
        print("已检测到现有索引，直接加载。")
        return load_vector_db()
    return build_index_from_data()


def print_sources(docs):
    print("\n—— 参考资料 ——")
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", "?")
        snippet = doc.page_content.strip().replace("\n", " ")[:80]
        print(f"[{i}] {os.path.basename(source)} (第{page}页): {snippet}...")


def interactive_loop(db):
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
        answer, docs = ask(question, db)
        print(f"\nAI> {answer}")
        print_sources(docs)


def main():
    rebuild = "--rebuild" in sys.argv
    # 收集非参数作为一次性问题
    question_args = [a for a in sys.argv[1:] if not a.startswith("--")]

    db = get_db(rebuild=rebuild)

    if question_args:
        question = " ".join(question_args)
        answer, docs = ask(question, db)
        print(f"问题：{question}")
        print(f"回答：{answer}")
        print_sources(docs)
    else:
        interactive_loop(db)


if __name__ == "__main__":
    main()
