"""端到端冒烟测试：分别验证 LLM、Embedding、FAISS、RAG 链是否可用。

运行：  python test_llm.py
"""
import os

from rag.vector_store import get_embedding, build_vector_db, load_vector_db
from rag.chain import get_llm, ask
from rag.loader import load_pdf, split_docs

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(PROJECT_ROOT, "data", "test.pdf")


def check_llm():
    print("\n[1] 测试 DeepSeek 对话模型 ...")
    llm = get_llm()
    reply = llm.invoke("用一句话回答：1+1 等于几？")
    print("    LLM 回复:", reply.content.strip())


def check_embedding():
    print("\n[2] 测试 Embedding 向量模型 ...")
    emb = get_embedding()
    vec = emb.embed_query("你好，向量测试")
    print(f"    向量维度: {len(vec)}, 前3值: {vec[:3]}")


def check_rag():
    print("\n[3] 测试完整 RAG 链路（检索 + 生成）...")
    if not os.path.exists(PDF_PATH):
        print("    跳过：未找到 data/test.pdf")
        return
    docs = load_pdf(PDF_PATH)
    chunks = split_docs(docs)
    build_vector_db(chunks)
    db = load_vector_db()
    question = "这份文档主要讲了什么内容？"
    answer, hits = ask(question, db)
    print(f"    问题: {question}")
    print(f"    命中文档数: {len(hits)}")
    print(f"    回答: {answer.strip()[:200]}")


if __name__ == "__main__":
    check_embedding()
    check_llm()
    check_rag()
    print("\n全部测试完成 ✅")
