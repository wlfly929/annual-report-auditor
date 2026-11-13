"""引用来源的统一结构：把「文档片段」转成 CLI / SSE / 评测都能消费的 dict。

为什么单独成模块：rag 产出 LangChain Document，agent 工具产出 dict，而 CLI、
SSE 接口、评测三处都要把两者归一成同一形状。这份转换之前抄在 main.py 与 api.py
各一份，评测为了复用还反向 import 了 CLI 入口。契约只放一处，就不会再漂移。
"""
from pathlib import Path

# 回传给前端的片段上限：前端只展示摘要，评测只比 (文件, 页码)，不需要全文
MAX_SOURCE_SNIPPET = 200


def make_source(file: str, page: int | str, snippet: str, max_chars: int = MAX_SOURCE_SNIPPET) -> dict:
    """构造一条引用。file 统一取 basename，避免把本机绝对路径带出去。"""
    return {
        "file": Path(str(file)).name or "未知文件",
        "page": page,
        "snippet": snippet[:max_chars],
    }


def docs_to_sources(docs) -> list[dict]:
    """把 LangChain Document 转成统一引用结构。

    向量库里的 page 是 0 基下标，展示与评测都按人类页码（1 基）比对。
    """
    result = []
    for doc in docs:
        page = doc.metadata.get("page")
        result.append(make_source(
            doc.metadata.get("source", ""),
            page + 1 if isinstance(page, int) else "?",
            doc.page_content.strip(),
        ))
    return result


def dedupe_sources(sources: list[dict]) -> list[dict]:
    """同一片段常被多轮检索重复命中，按 (文件, 页码) 去重后保留首次出现顺序。"""
    seen, unique = set(), []
    for item in sources:
        key = (item["file"], item["page"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique
