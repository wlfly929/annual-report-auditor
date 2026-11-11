"""记录已入库 PDF 的登记簿，用于增量更新索引。

结构: {文件内容sha256: {"source": 文件名, "ids": [chunk_id, ...]}}
登记簿与索引同放于 faiss_index 目录，删掉索引即等同重新建库。
"""
import hashlib
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "faiss_index", "manifest.json")
# 索引参数指纹单独存放。不能塞进 manifest：那个 dict 的每个 key 都被当作文件哈希遍历
META_PATH = os.path.join(PROJECT_ROOT, "faiss_index", "index_meta.json")


def file_hash(path: str) -> str:
    """按文件内容算哈希，内容变了哈希就变，与修改时间无关。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def index_signature() -> str:
    """把影响切块结果的参数揉成一个指纹。

    只按文件哈希判断“索引是否最新”有个隐蔽缺口：改了 chunk_size 而 PDF 没变时，
    程序会认为索引是最新的，继续用旧索引，表现就是“参数改了但指标纹丝不动”。
    """
    # 延迟导入：避免 manifest 与 loader 在模块层互相引用
    from rag.loader import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_CHARS, MIN_PAGE_CHARS
    embedding = os.getenv("EMBEDDING_MODEL", "")
    raw = f"{CHUNK_SIZE}|{CHUNK_OVERLAP}|{MIN_PAGE_CHARS}|{MIN_CHUNK_CHARS}|{embedding}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_index_meta() -> dict:
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_index_meta(meta: dict):
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
