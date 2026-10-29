import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 基于文件位置定位项目根目录，确保 .env 与索引路径不受运行时工作目录影响
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# 默认索引存放位置（项目根目录下的 faiss_index 文件夹）
DEFAULT_INDEX_PATH = os.path.join(PROJECT_ROOT, "faiss_index")


def get_embedding():
    """构造 Embedding（走 OpenAI 兼容接口，DeepSeek 不提供向量模型）。

    兼容任意 OpenAI 格式端点：
      - OPENAI_API_KEY   必填
      - OPENAI_BASE_URL  选填，不填则用 OpenAI 官方端点
      - EMBEDDING_MODEL  选填，不同端点模型名不同
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未检测到 OPENAI_API_KEY，请保存 .env 后重试（注意别只在编辑器里改而未落盘）"
        )
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    kwargs = {
        "openai_api_key": api_key,
        "model": model,
        # 非 OpenAI 官方端点（如通义）需关掉 tiktoken 预分词，否则报 input 类型错误
        "check_embedding_ctx_length": False,
    }
    # 只有配置了自定义端点才传入，否则默认连 OpenAI 官方
    if base_url:
        kwargs["openai_api_base"] = base_url
    return OpenAIEmbeddings(**kwargs)


def build_vector_db(chunks, save_path=DEFAULT_INDEX_PATH, ids=None):
    """新建向量库；传入 ids 时使用稳定 ID，便于后续按文件精准删除。"""
    embeddings = get_embedding()
    if ids is not None:
        db = FAISS.from_documents(chunks, embeddings, ids=ids)
    else:
        db = FAISS.from_documents(chunks, embeddings)
    db.save_local(save_path)
    return db


def load_vector_db(save_path=DEFAULT_INDEX_PATH):
    embeddings = get_embedding()
    db = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    return db


def add_to_vector_db(db, new_chunks, ids):
    """把新块追加进已有索引，不重算旧向量；用稳定 id 便于后续删除。"""
    db.add_documents(new_chunks, ids=ids)
    return db


def delete_from_vector_db(db, ids):
    """按稳定 ID 删除某个文件的历史向量（FAISS 标记删除）。"""
    db.delete(ids)
    return db


def save_vector_db(db, save_path=DEFAULT_INDEX_PATH):
    db.save_local(save_path)
    return db
