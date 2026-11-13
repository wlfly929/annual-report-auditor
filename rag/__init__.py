"""检索基础设施：加载、切块、向量库、增量登记簿。

本包刻意不依赖 agent/，可被独立复用。对外导出需要什么就直接用子模块导入：

    from rag.indexer import index_documents
    from rag.loader import extract_pages

早期这里维护过一份 __all__ 聚合清单，但全项目没有一处 `from rag import ...`，
维护清单反而成了额外负担，故移除。
"""
