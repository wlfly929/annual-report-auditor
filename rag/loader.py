from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "，", " "]
    )
    chunks = splitter.split_documents(docs)
    return chunks

if __name__ == "__main__":
    # 基于项目根目录拼绝对路径，避免受运行时工作目录影响
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = os.path.join(project_root, "data", "test.pdf")
    print(f"查找路径: {pdf_path}")
    if os.path.exists(pdf_path):
        documents = load_pdf(pdf_path)
        chunks = split_docs(documents)
        print(f"一共切分 {len(chunks)} 块")
    else:
        print("请先在data文件夹放入test.pdf")