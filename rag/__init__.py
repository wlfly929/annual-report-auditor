"""RAG 项目对外统一接口。"""
from .loader import load_pdf, split_docs
from .vector_store import (
    get_embedding,
    build_vector_db,
    load_vector_db,
    add_to_vector_db,
    delete_from_vector_db,
    save_vector_db,
    DEFAULT_INDEX_PATH,
)
from .manifest import file_hash, load_manifest, save_manifest
from .chain import get_llm, build_rag_chain, ask

__all__ = [
    "load_pdf",
    "split_docs",
    "get_embedding",
    "build_vector_db",
    "load_vector_db",
    "add_to_vector_db",
    "delete_from_vector_db",
    "save_vector_db",
    "DEFAULT_INDEX_PATH",
    "file_hash",
    "load_manifest",
    "save_manifest",
    "get_llm",
    "build_rag_chain",
    "ask",
]
