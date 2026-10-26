"""RAG 项目对外统一接口。"""
from .loader import load_pdf, split_docs
from .vector_store import (
    get_embedding,
    build_vector_db,
    load_vector_db,
    DEFAULT_INDEX_PATH,
)
from .chain import get_llm, build_rag_chain, ask

__all__ = [
    "load_pdf",
    "split_docs",
    "get_embedding",
    "build_vector_db",
    "load_vector_db",
    "DEFAULT_INDEX_PATH",
    "get_llm",
    "build_rag_chain",
    "ask",
]
