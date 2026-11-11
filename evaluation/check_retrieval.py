"""检索层评估：只跑向量召回，不调 LLM。

为什么必须单独测检索层：
    报告里的"页码命中率"取自模型答案的引用，混了生成端噪声。更隐蔽的是
    "页码命中但内容不对"——一页被切成多块后，检索只召回其中语义最相近的那块，
    页码对得上，但答案所在的那块根本没进上下文。只看页码会把这个漏掉。
    实测：问营业收入时召回的是 p7 的"其他有关资料"块，而营业收入表在 p7 的
    另一块里，页码命中、答案却不在。

    所以这里同时统计两个指标：
        页码命中  目标页是否进入 top-k        → 衡量"覆盖范围"
        要点覆盖  答案要点是否出现在召回文本 → 衡量"内容完整性"

用法：
    python evaluation/check_retrieval.py             # 默认测 k=4,6,8,10
    python evaluation/check_retrieval.py --k 4 8 12
"""
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.questions import QUESTIONS  # noqa: E402
from evaluation.run_eval import normalize  # noqa: E402
from main import index_documents  # noqa: E402
from rag.loader import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_CHARS, MIN_PAGE_CHARS  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, nargs="+", default=[4, 6, 8, 10])
    parser.add_argument("--out", default=str(PROJECT_ROOT / "evaluation" / "retrieval.md"))
    args = parser.parse_args()

    db = index_documents()
    total_chunks = len(db.index_to_docstore_id)
    answerable = [q for q in QUESTIONS if not q["refuse"]]
    # 计算题的要点是算出来的值，原文里必然搜不到，只统计其余题目
    point_items = [q for q in answerable if not q.get("computed")]

    max_k = max(args.k)
    # 每题只检索一次（取最大 k），再按不同 k 截断，保证各组比较的是同一批召回
    cache = {}
    for q in answerable:
        docs = db.similarity_search(q["question"], k=max_k)
        cache[q["id"]] = [
            (os.path.basename(d.metadata["source"]), d.metadata["page"] + 1,
             d.page_content)
            for d in docs
        ]

    lines = [
        "# 检索层评估",
        "",
        f"- 入库策略：chunk_size={CHUNK_SIZE} / overlap={CHUNK_OVERLAP}；"
        f"丢弃 <{MIN_PAGE_CHARS} 字的页与 <{MIN_CHUNK_CHARS} 字的块",
        f"- 当前索引：{total_chunks} 块",
        f"- 可答题：{len(answerable)} 条（其中计算题 {len(answerable) - len(point_items)} 条不计入要点覆盖）",
        "",
        "| k | 页码命中 | 要点覆盖 |",
        "|---|---|---|",
    ]

    details = []
    for k in args.k:
        page_hit = 0
        point_sum = 0.0
        miss_point = []
        for q in answerable:
            docs = cache[q["id"]][:k]
            keys = {(f, p) for f, p, _ in docs}
            if all(any((f, p) in keys for p in q["expect_pages"]) for f in q["expect_files"]):
                page_hit += 1

            if q.get("computed"):
                continue
            corpus = normalize("".join(t for _, _, t in docs))
            groups = q["expect_points"]
            hit = [i for i, g in enumerate(groups, 1)
                   if any(normalize(key) in corpus for key in g)]
            point_sum += len(hit) / len(groups)
            if k == max_k and len(hit) < len(groups):
                miss_point.append(f"{q['id']} 缺要点组 {sorted(set(range(1, len(groups) + 1)) - set(hit))}")

        n = len(answerable)
        np_ = len(point_items)
        lines.append(
            f"| {k} | {page_hit}/{n} ({page_hit / n * 100:.1f}%) "
            f"| {point_sum / np_ * 100:.1f}% |"
        )
        details.extend(miss_point)

    if details:
        lines += ["", f"## k={max_k} 时要点仍未覆盖的题", ""]
        lines += [f"- {d}" for d in details]

    # 目标页排名：看失败题到底是“差一点”还是“完全不在候选里”，
    # 前者提 k 能救，后者只能改检索策略（混合检索 / 换查询词）
    lines += ["", f"## 目标页在 top-{max_k} 里的最佳排名", "",
              "| ID | 预期页 | 各文件排名 | 最差 |", "|---|---|---|---|"]
    for q in answerable:
        seq = [(f, p) for f, p, _ in cache[q["id"]]]
        per_file = []
        for f in q["expect_files"]:
            per_file.append(next(
                (i + 1 for i, (ff, pp) in enumerate(seq)
                 if ff == f and pp in q["expect_pages"]), None))
        worst = None if any(r is None for r in per_file) else max(per_file)
        lines.append(
            f"| {q['id']} | {','.join(str(p) for p in q['expect_pages'])} "
            f"| {per_file} | {worst or f'均未进 top-{max_k}'} |"
        )

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入 {args.out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
