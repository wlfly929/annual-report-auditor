"""PDF 提取与切块。

提取器用 PyMuPDF 而不是 pypdf：港股年报等文档的中文字体缺少 ToUnicode 映射，
pypdf 会输出乱码（实测腾讯年报读出来是"ಥ࿆ᓞٰ̨΅˾໮"），PyMuPDF 能正确还原。
"""
import collections
import os

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 跨页面重复率达到该比例的行视为页眉/页脚，入库前剔除。
# 不清理的话每个 chunk 都带着同一串页眉，会让所有块的向量互相趋同，检索区分度下降。
HEADER_FOOTER_PAGE_RATIO = 0.8
# 只有短行才可能是页眉页脚；正文长句不参与统计，避免误删
HEADER_FOOTER_MAX_LEN = 60
HEADER_FOOTER_MIN_LEN = 4

# 切块粒度：与“按页完整”对齐，而不是切得越细越好。
# 实测把 chunk_size 降到 350 是负优化：同一页被切成 3~4 块后，检索只召回
# 语义最相近的那一块。问营业收入时召回的是 p7 的“其他有关资料”块，营业
# 收入表则在 p7 的另一块里——页码命中、答案却不在，要点覆盖卡在 71.9%，
# 而 k 从 6 加到 10 也救不回来（答案块压根进不了候选）。
# 800 能保住一页的完整表格（苏宁 p7 清理后 794 字，刚好一整块），
# 同时把超长页拆开；代价就是每页 1~2 块，所以 k 要相应调大（见 config）。
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# 清理页眉页脚后不足该长度的页直接不入库：封面、章节隔页、签字页只有几十字，
# 但实体词密度极高（如“苏宁环球股份有限公司 2025 年年度报告”），对任何查询都会
# 给出中等相似度。实测 33 字的封面页在“分地区收入”查询里排第 1，把真正的答案页
# 挤出了 top-8。注意这只影响入库，extract_pages 仍能读到这些页，精读不受影响。
MIN_PAGE_CHARS = 80
# 切块后同样丢弃过短块：页尾残句、目录条目单独成块时，和任何问题都“有点像”，
# 只会白占 top-k 名额
MIN_CHUNK_CHARS = 80


def _find_header_footer_lines(pages):
    """找出在大多数页面里重复出现的短行（页眉、页脚、连续财报的表头）。"""
    if len(pages) < 3:
        return set()
    counter = collections.Counter()
    for text in pages:
        # 同一页内先用 set 去重，否则单页内重复的行会虚增计数
        for line in {l.strip() for l in text.splitlines()}:
            if HEADER_FOOTER_MIN_LEN <= len(line) <= HEADER_FOOTER_MAX_LEN:
                counter[line] += 1
    threshold = len(pages) * HEADER_FOOTER_PAGE_RATIO
    return {line for line, count in counter.items() if count >= threshold}


def load_pdf(file_path: str):
    """按页提取文本，返回 Document 列表（metadata.page 为 0-based 原始页码）。"""
    with pymupdf.open(file_path) as doc:
        pages = [page.get_text().strip() for page in doc]

    drop = _find_header_footer_lines(pages)
    documents = []
    for index, text in enumerate(pages):
        if drop:
            text = "\n".join(
                line for line in text.splitlines() if line.strip() not in drop
            ).strip()
        if len(text) < MIN_PAGE_CHARS:
            continue
        # 页码取原始下标，跳过的空页不能导致后续页码整体偏移
        documents.append(Document(
            page_content=text,
            metadata={"source": file_path, "page": index},
        ))
    return documents


def extract_pages(file_path: str, start_page: int = 1, end_page: int = None):
    """提取指定页码范围的文本，返回 ({页码: 文本}, 总页数)；页码 1-based 且含首尾。

    与 load_pdf 共用同一套页眉页脚清理规则。Agent 的 read_document 走这里，
    保证“检索看到的片段”与“精读看到的原文”是同一份干净文本；
    两边用不同提取器会出现检索正常、精读拿到乱码的怪现象。
    """
    with pymupdf.open(file_path) as doc:
        total = doc.page_count
        raw = [page.get_text().strip() for page in doc]

    drop = _find_header_footer_lines(raw)
    first = max(1, start_page)
    last = min(end_page or total, total)

    pages = {}
    for page_no in range(first, last + 1):
        text = raw[page_no - 1]
        if drop:
            text = "\n".join(
                line for line in text.splitlines() if line.strip() not in drop
            ).strip()
        pages[page_no] = text
    return pages, total


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " "],
    )
    chunks = splitter.split_documents(docs)
    return [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_CHARS]

if __name__ == "__main__":
    # 基于项目根目录拼绝对路径，避免受运行时工作目录影响
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    pdfs = sorted(f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")) if os.path.isdir(data_dir) else []
    if not pdfs:
        print(f"请先在 {data_dir} 放入 PDF 文件")
    for name in pdfs:
        pdf_path = os.path.join(data_dir, name)
        docs = load_pdf(pdf_path)
        chunks = split_docs(docs)
        chars = sum(len(d.page_content) for d in docs)
        lens = [len(c.page_content) for c in chunks]
        avg = sum(lens) / len(lens) if lens else 0
        print(f"{name}: 入库 {len(docs)} 页 / {chars} 字 -> 切分 {len(chunks)} 块"
              f"（均长 {avg:.0f} 字，最短 {min(lens) if lens else 0} 字）")