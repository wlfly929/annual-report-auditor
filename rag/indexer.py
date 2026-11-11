"""索引构建与增量维护。

从 main.py 抽出来：索引管理不是 CLI 的职责。服务层（api.py）启动时同样要
加载并校验索引，如果这段逻辑只留在入口脚本里，API 就得反过来 import CLI，
职责倒挂，链路也会因为多一层 import 而变脆。
"""
import os

from config import settings
from rag.loader import load_pdf, split_docs
from rag.manifest import (
    file_hash,
    index_signature,
    load_index_meta,
    load_manifest,
    save_index_meta,
    save_manifest,
)
from rag.vector_store import (
    DEFAULT_INDEX_PATH,
    add_to_vector_db,
    build_vector_db,
    delete_from_vector_db,
    load_vector_db,
    save_vector_db,
)

DATA_DIR = str(settings.data_dir)


def scan_pdf_files():
    os.makedirs(DATA_DIR, exist_ok=True)     # 首次运行时 data 目录可能还不存在
    result = {}
    for f in os.listdir(DATA_DIR):          # 列出 data 目录下所有文件名
        if f.lower().endswith(".pdf"):      # 只要 .pdf 结尾的（忽略大小写）
            path = os.path.join(DATA_DIR, f)             # 拼出完整路径
            h = file_hash(path)                          # 算这个文件内容的 sha256
            result[h] = path                             # 哈希做键、路径做值
    return result


def index_documents(force_rebuild: bool = False):
    """增量维护索引：只对新 PDF / 内容变化的 PDF 做切块和向量化。"""
    manifest = {} if force_rebuild else load_manifest()
    current = scan_pdf_files()
    if not current:
        raise FileNotFoundError(f"{DATA_DIR} 下没有找到任何 PDF 文件")

    index_exists = os.path.exists(os.path.join(DEFAULT_INDEX_PATH, "index.faiss"))

    # 切块参数变了、PDF 却没变时，光看 manifest 会误判“索引已是最新”，
    # 结果是改了参数却一直在用旧索引，指标看上去纹丝不动。用参数指纹兜住这个缺口。
    signature = index_signature()
    if index_exists and load_index_meta().get("signature") != signature:
        if not force_rebuild:
            print("切块/向量化参数已变更，改为全量重建索引。")
        force_rebuild, manifest = True, {}

    # 需要删除的：登记过但文件已被移除（内容变化的文件哈希会变，等同于旧哈希被删除）
    to_delete = {h: info for h, info in manifest.items() if h not in current}
    # 需要新增的：没登记过的文件（新文件或修改后的文件）
    to_add = [p for h, p in current.items() if h not in manifest]

    if not to_delete and not to_add and index_exists:
        print("索引已是最新，直接加载。")
        return load_vector_db()

    # 全量重建时不加载旧库，从头构建；否则有旧库就加载以便追加/删除
    db = load_vector_db() if (index_exists and not force_rebuild) else None

    # 1. 删除已失效文件的向量（稳定 ID 保证可精准删除）
    for h, info in to_delete.items():
        if db is not None and info.get("ids"):
            delete_from_vector_db(db, info["ids"])
        print(f"已移除旧向量: {info['source']}")
        manifest.pop(h, None)

    # 2. 只对新增/变更文件切块 + 向量化
    for path in to_add:
        name = os.path.basename(path)
        h = file_hash(path)
        print(f"正在索引新文件: {name}")
        chunks = split_docs(load_pdf(path))
        # 稳定 ID：文件哈希 + 块序号，同一内容永远同一 ID，便于后续删除
        ids = [f"{h}_{i}" for i in range(len(chunks))]
        if db is None:
            db = build_vector_db(chunks, ids=ids)
        else:
            add_to_vector_db(db, chunks, ids)
        manifest[h] = {"source": name, "ids": ids}
        print(f"  -> 新增 {len(chunks)} 块（其他文件向量未重算）")

    # 3. 持久化索引与登记簿
    save_vector_db(db)
    save_manifest(manifest)
    save_index_meta({"signature": signature})
    print(f"更新完成，当前共登记 {len(manifest)} 个文件。")
    return db
