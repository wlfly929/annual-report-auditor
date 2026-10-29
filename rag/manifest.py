"""记录已入库 PDF 的登记簿，用于增量更新索引。

结构: {文件内容sha256: {"source": 文件名, "ids": [chunk_id, ...]}}
登记簿与索引同放于 faiss_index 目录，删掉索引即等同重新建库。
"""
import hashlib
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "faiss_index", "manifest.json")


def file_hash(path: str) -> str:
    """按文件内容算哈希，内容变了哈希就变，与修改时间无关。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
