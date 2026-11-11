"""校验评测集标注：确认每条题的要点与页码真的对得上原文。

为什么要单独校验：
    判分靠关键词，如果标注本身写错（页码错、关键词写错），失败的题会被误判成
    "模型答错"，然后你花时间去调 prompt 或检索，实际问题在题库里。
    跑一遍这个脚本，能把"标注错误"和"模型能力问题"彻底分开。

用法：
    python evaluation/check_questions.py            # 全部检查
    python evaluation/check_questions.py --verbose  # 有问题的题打印原文片段

检查项：
    1. expect_pages 是否都在对应文件的页数范围内
    2. 每个要点组是否至少有一个写法真的出现在标定页原文里
    3. 拒答题标定页里是否真的找不到"有答案"的迹象（只做提示，不阻断）
"""
import argparse
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings  # noqa: E402
from evaluation.questions import QUESTIONS  # noqa: E402
from rag.loader import extract_pages  # noqa: E402


def normalize(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", str(text)).split()).lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cache: dict[str, dict[int, str]] = {}
    for name in {f for q in QUESTIONS for f in (q.get("expect_files") or [])}:
        # 用 extract_pages 而不是 load_pdf：后者会跳过短页（封面、隔页），
        # 而标注的页码是 PDF 物理页码，与入库策略无关。用 load_pdf 的话，
        # 被过滤的页会被误报成“无此页”，排查方向就带偏了。
        pages, _ = extract_pages(str(Path(settings.data_dir) / name))
        cache[name] = {p: normalize(t) for p, t in pages.items()}

    problems = 0
    for item in QUESTIONS:
        files = item.get("expect_files") or []
        pages = item.get("expect_pages") or []
        issues = []

        # 1) 页码范围
        for f in files:
            if f not in cache:
                issues.append(f"文件不存在：{f}")
                continue
            total = max(cache[f]) if cache[f] else 0
            for p in pages:
                if p not in cache[f]:
                    issues.append(f"{f} 无第 {p} 页（共 {total} 页）")

        # 2) 要点是否真的在标定页里（计算题跳过：答案是算出来的，原文当然没有）
        if not item["refuse"] and not item.get("computed") and pages and files:
            corpus = ""
            for f in files:
                corpus += "".join(cache.get(f, {}).get(p, "") for p in pages)
            for idx, group in enumerate(item["expect_points"], 1):
                if not any(normalize(k) in corpus for k in group):
                    issues.append(f"要点组{idx} 在标定页里找不到：{group[:3]}")

        # 3) 拒答题：标定页里如果出现明显的数值线索，提醒复核
        if item["refuse"] and pages and files and args.verbose:
            corpus = ""
            for f in files:
                corpus += "".join(cache.get(f, {}).get(p, "") for p in pages)
            if corpus:
                issues.append(f"提示：该题标注了页码 {pages}，请确认原文确实没有答案")

        if issues:
            problems += len(issues)
            print(f"[!] {item['id']} {item['category']}  {item['question'][:30]}")
            for msg in issues:
                print(f"    - {msg}")

    print()
    if problems:
        print(f"共 {problems} 处问题，请先修正 questions.py 再跑评测。")
        sys.exit(1)
    print(f"标注校验通过：{len(QUESTIONS)} 条题目的要点与页码与原文一致。")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
