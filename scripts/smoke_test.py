"""端到端冒烟测试：分别验证 LLM、Embedding、FAISS、RAG 链是否可用。

运行：  python scripts/smoke_test.py

注意：前两项会真实调用 API；第三项需要 data/ 下至少有一个 PDF，没有则跳过。
"""
import sys
import tempfile
from pathlib import Path

# 允许 `python scripts/smoke_test.py` 直接运行（此时 sys.path[0] 是 scripts/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.vector_store import get_embedding, build_vector_db, load_vector_db  # noqa: E402
from rag.chain import get_llm, ask  # noqa: E402
from rag.loader import load_pdf, split_docs  # noqa: E402


def first_pdf():
    """取 data/ 下第一个 PDF。不硬编码文件名：换语料时脚本不应该随之失效。"""
    pdfs = sorted((PROJECT_ROOT / "data").glob("*.pdf"))
    return pdfs[0] if pdfs else None


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
    pdf_path = first_pdf()
    if pdf_path is None:
        print("    跳过：data/ 下没有 PDF")
        return
    docs = load_pdf(str(pdf_path))
    chunks = split_docs(docs)
    # 必须写到临时目录：build_vector_db 内部会 save_local，
    # 用默认路径会把正式索引覆盖成"只有这一个 PDF"的残库。
    with tempfile.TemporaryDirectory() as tmp:
        build_vector_db(chunks, save_path=tmp)
        db = load_vector_db(save_path=tmp)
        question = "这份文档主要讲了什么内容？"
        answer, hits = ask(question, db)
    print(f"    语料: {pdf_path.name}")
    print(f"    问题: {question}")
    print(f"    命中文档数: {len(hits)}")
    print(f"    回答: {answer.strip()[:200]}")


if __name__ == "__main__":
    check_embedding()
    check_llm()
    check_rag()
    print("\n全部测试完成 ✅")
