"""Agent 的四个工具。

工具用**闭包工厂**构造：
    db        —— 向量库实例，由调用方注入，便于测试时替换
    collected —— 引用收集列表，由调用方持有，用于把溯源信息回传给界面

不写成模块级单例的原因：后续接 Web 服务时，每个会话需要独立的引用记录，
闭包天然按会话隔离，不会互相污染（当前 CLI 单会话下无此问题）。
"""
import ast
import os
import re
from datetime import datetime

from langchain_core.tools import tool

from config import settings
from rag.loader import extract_pages
from sources import make_source

# 单个片段回传给模型的最大字符数，控制上下文预算
MAX_SNIPPET_CHARS = 600
# 单次精读的页数上限，防止一口气读几十页把上下文撑爆
MAX_READ_PAGES = 10

# calculate 的运算白名单：只放行四则运算，**不用 eval**，避免任意代码执行
_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}
_UNARY_OPS = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}


def _eval_node(node):
    """递归求值 AST，只接受常量与白名单运算符。"""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("表达式只支持数字与 + - * / // % ** 运算")


def _safe_calculate(expression: str) -> float:
    return _eval_node(ast.parse(expression.strip(), mode="eval"))


def _sanitize_filename(name: str) -> str:
    """去掉 Windows 非法字符，避免 write_report 因标题里的冒号/斜杠写盘失败。"""
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", name).strip("_")
    return cleaned[:50] or "report"


def _human_page(metadata: dict) -> int | str:
    """向量库里的 page 是 0 基下标，转成人类可读的页码。"""
    page = metadata.get("page")
    return page + 1 if isinstance(page, int) else "?"


def make_tools(db, collected: list[dict]):
    """构造工具集。collected 由调用方持有，工具内部直接 append（单会话假设）。"""

    @tool
    def search_knowledge_base(query: str, top_k: int = settings.retrieval_top_k) -> str:
        """在本地文档库中做语义检索，返回最相关的若干片段（含文件名与页码）。

        这是"粗筛"：用于快速定位线索。若返回内容不足以回答，可换关键词再查一次。

        Args:
            query: 检索意图，用具体的名词短语或完整问句，避免"这个""上面说的"这类指代。
            top_k: 返回片段数量，默认 8（上限 10）；需要更广覆盖时可调高。
        """
        k = max(1, min(int(top_k), 10))
        docs = db.similarity_search(query, k=k)
        if not docs:
            return "检索结果为空：文档库中没有与该查询相关的内容。"

        blocks = []
        for i, doc in enumerate(docs, start=1):
            source = os.path.basename(doc.metadata.get("source", "未知文件"))
            page = _human_page(doc.metadata)
            snippet = doc.page_content.strip()[:MAX_SNIPPET_CHARS]
            collected.append(make_source(source, page, snippet, max_chars=MAX_SNIPPET_CHARS))
            blocks.append(f"[片段{i}] {source} p.{page}\n{snippet}")
        return "\n\n".join(blocks)

    @tool
    def read_document(file_name: str, start_page: int = 1, end_page: int = 1) -> str:
        """读取指定 PDF 的指定页原文（精读），页码从 1 开始，包含首尾两页。

        用于核对检索片段被截断的上下文、确认数字口径，或检索定位不到时翻原页。

        Args:
            file_name: data 目录下的文件名，例如 "2024年报.pdf"。
            start_page: 起始页，从 1 开始。
            end_page: 结束页（含）。只读一页时与 start_page 相同。
        """
        # 只取 basename，防止模型拼出 ../ 之类的路径穿越
        path = settings.data_dir / os.path.basename(file_name)
        if not path.exists():
            available = sorted(p.name for p in settings.data_dir.glob("*.pdf"))
            return f"未找到文件 {file_name}。当前可读文件：{available or '（data 目录为空）'}"

        start = max(1, int(start_page))
        end = max(start, int(end_page))
        if end - start + 1 > MAX_READ_PAGES:
            end = start + MAX_READ_PAGES - 1

        try:
            # 用与索引入库同一套提取器（PyMuPDF + 页眉清理），避免精读拿到乱码
            pages, total = extract_pages(str(path), start, end)
        except Exception as exc:  # 加密、损坏、非标准 PDF
            return f"读取 {file_name} 失败：{exc}"
        end = min(end, total)

        texts = []
        for page_no in range(start, end + 1):
            text = pages.get(page_no, "")
            # 精读也是来源：不记录的话，评测里的"检索覆盖"会低估 Agent 的实际取数范围
            collected.append(make_source(
                path.name, page_no, text, max_chars=MAX_SNIPPET_CHARS
            ))
            texts.append(f"--- {path.name} p.{page_no} ---\n{text or '(本页无可提取文本，可能是扫描件)'}")

        if not texts:
            return f"{path.name} 页码范围无效（该文件共 {total} 页）。"
        return "\n\n".join(texts)

    @tool
    def calculate(expression: str) -> str:
        """计算数学表达式，支持 + - * / // % ** 与括号。

        凡是涉及加减乘除的结论都必须调用本工具，不要心算，避免数字错误被写进报告。

        Args:
            expression: 纯数字表达式，例如 "(1200-800)/800*100"。
        """
        try:
            return f"{expression} = {_safe_calculate(expression):g}"
        except ZeroDivisionError:
            return "计算失败：除数为 0。"
        except Exception as exc:
            return f"计算失败：{exc}"

    @tool
    def write_report(title: str, content: str) -> str:
        """把研究报告写入 reports/ 目录（Markdown），返回文件路径。

        仅当用户明确要求"报告 / 分析文档 / 总结成文件"时调用；
        普通问答直接把答案返回给用户即可，不要落盘。

        Args:
            title: 报告标题，会用作文件名的一部分。
            content: 报告正文，支持 Markdown，应包含结论与来源引用。
        """
        settings.report_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = settings.report_dir / f"{stamp}_{_sanitize_filename(title)}.md"
        path.write_text(content, encoding="utf-8")
        return f"报告已写入：{path}"

    return [search_knowledge_base, read_document, calculate, write_report]
