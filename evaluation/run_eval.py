"""评测执行器：同一套题目分别跑 baseline（固定管线）与 Agent，输出对照表。

用法：
    python evaluation/run_eval.py                  # 全量 18 条，两种模式
    python evaluation/run_eval.py --limit 8        # 按类别轮流取 8 条，先验证流程
    python evaluation/run_eval.py --mode agent     # 只跑 Agent

评分说明：
    不用 LLM 当裁判。LLM-as-judge 会引入二次不确定性，且每题要多花一次调用；
    这里用「要点关键词覆盖 + 页码命中」做确定性判分，同一条问题反复跑结果可比。
"""
import argparse
import json
import os
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

# 允许 `python evaluation/run_eval.py` 直接运行（此时 sys.path[0] 是 evaluation/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from agent.graph import build_agent  # noqa: E402
from config import settings  # noqa: E402
from evaluation.questions import QUESTIONS  # noqa: E402
from main import docs_to_sources, index_documents  # noqa: E402
from rag.chain import RAG_PROMPT, get_llm  # noqa: E402
from rag.loader import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_CHARS, MIN_PAGE_CHARS  # noqa: E402

REFUSAL_MARKERS = [
    # 基础否定词放最前：中文否定表达极多（没找到 / 无法回答 / 未收录…），
    # 逐条枚举必然漏，命中"没有 / 未 / 无法"这类基词即可判定为拒答。
    "没有", "未", "无相关", "无法", "不存在", "缺少", "未知", "不涉及", "不包含",
]


def normalize(text: str) -> str:
    """去掉空白、统一大小写与全半角，避免格式差异影响判分。

    NFKC 把全角百分号、全角数字折算成半角，否则"13／87亿"与"13.87亿"、
    "60％"与"60%"会被当成不同字符串。注意 NFKC 不做繁简转换，繁体要点需在
    questions.py 里单独列出。
    """
    return "".join(unicodedata.normalize("NFKC", str(text)).split()).lower()


def source_keys(sources: list[dict]) -> set:
    """把引用来源转成 (文件名, 页码) 集合。

    多文档下页码会撞车（两份年报都有第 7 页），只比页码会把"查错文档但页码对了"
    误判为命中，所以必须成对比较。
    """
    keys = set()
    for s in sources:
        file_name = os.path.basename(str(s.get("file", "")))
        keys.add((file_name, str(s.get("page"))))
    return keys


def sources_ok(sources: list[dict], item: dict) -> bool:
    expected_files = item.get("expect_files") or []
    expected_pages = item.get("expect_pages") or []
    if not expected_files or not expected_pages:
        return True

    keys = source_keys(sources)
    pages = [str(p) for p in expected_pages]
    # 单文档：命中任意一条预期页即可
    if len(expected_files) == 1:
        return any((expected_files[0], p) in keys for p in pages)
    # 跨文档：每个文档都要至少命中一页，只查一半不算完成
    return all(any((f, p) in keys for p in pages) for f in expected_files)


def format_docs(docs) -> str:
    return "\n\n".join(f"[资料{i}] {d.page_content.strip()}" for i, d in enumerate(docs, 1))


def score(answer: str, sources: list[dict], item: dict) -> dict:
    text = normalize(answer)
    groups = item["expect_points"]
    hit = sum(1 for g in groups if any(normalize(k) in text for k in g))
    point_rate = hit / len(groups) if groups else 1.0

    if item["refuse"]:
        markers = REFUSAL_MARKERS + item.get("refuse_allow", [])
        refused = any(normalize(m) in text for m in markers)
        return {"point_rate": 1.0 if refused else 0.0, "page_hit": None, "passed": refused}

    page_hit = sources_ok(sources, item)
    return {"point_rate": point_rate, "page_hit": page_hit, "passed": point_rate >= 0.6 and page_hit}


def run_agent(agent, collected: list[dict], question: str, thread_id: str) -> dict:
    collected.clear()
    payload = {"messages": [HumanMessage(content=question)], "question": question, "steps": 0}
    cfg = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.agent_max_steps * 2 + 5,
    }

    steps = tool_calls = in_tok = out_tok = 0
    answer = "(未产出答案)"
    start = time.perf_counter()
    for chunk in agent.stream(payload, config=cfg, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "agent":
                steps += 1
                msg = update["messages"][-1]
                usage = getattr(msg, "usage_metadata", None) or {}
                in_tok += usage.get("input_tokens", 0)
                out_tok += usage.get("output_tokens", 0)
                if not (getattr(msg, "tool_calls", None) or []):
                    answer = msg.content
            elif node == "tools":
                tool_calls += len(update["messages"])
            elif node == "guard":
                for msg in update["messages"]:
                    if isinstance(msg, AIMessage):
                        answer = msg.content
    latency = time.perf_counter() - start

    return {
        "answer": answer,
        "sources": list(collected),
        "steps": steps,
        "tool_calls": tool_calls,
        "latency": latency,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }


def pick_round_robin(questions: list[dict], limit: int) -> list[dict]:
    """按类别轮流取题，保证 --limit 小样本时四类都被覆盖。"""
    if limit <= 0:
        return questions
    buckets: dict[str, list[dict]] = {}
    for q in questions:
        buckets.setdefault(q["category"], []).append(q)

    picked, depth = [], 0
    while len(picked) < limit:
        added = False
        for items in buckets.values():
            if depth < len(items) and len(picked) < limit:
                picked.append(items[depth])
                added = True
        if not added:
            break
        depth += 1
    return picked


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def median(values: list[float]) -> float:
    """延迟是长尾分布：个别请求卡在重试上能把平均值拉高一个数量级（实测有一条 811s
    的请求把 30 条的平均值从 1s 抬到 28s），所以延迟主指标用中位数。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def build_summary(rows: list[dict], mode: str) -> dict:
    valid = [r for r in rows if r[mode].get("error") is None]
    n = len(valid) or 1
    answerable = [r for r in valid if not r["refuse"]]
    refused = [r for r in valid if r["refuse"]]

    return {
        "通过率": f"{sum(1 for r in valid if r[mode]['score']['passed'])}/{len(valid)}",
        "要点覆盖率": sum(r[mode]["score"]["point_rate"] for r in answerable) / (len(answerable) or 1),
        "页码命中率": sum(1 for r in answerable if r[mode]["score"]["page_hit"]) / (len(answerable) or 1),
        "拒答正确率": f"{sum(1 for r in refused if r[mode]['score']['passed'])}/{len(refused) or 1}",
        "平均步数": sum(r[mode]["steps"] for r in valid) / n,
        "平均工具调用": sum(r[mode]["tool_calls"] for r in valid) / n,
        "平均延迟": sum(r[mode]["latency"] for r in valid) / n,
        "延迟中位数": median([r[mode]["latency"] for r in valid]),
        "平均输入token": sum(r[mode]["input_tokens"] for r in valid) / n,
        "平均输出token": sum(r[mode]["output_tokens"] for r in valid) / n,
    }


def describe_corpus() -> str:
    """列出本次评测的语料（文件 + 页数），让报告能自我说明。"""
    import pymupdf

    data_dir = Path(settings.data_dir)
    parts = []
    for pdf in sorted(data_dir.glob("*.pdf")):
        with pymupdf.open(str(pdf)) as doc:
            parts.append(f"`{pdf.name}`（{doc.page_count} 页）")
    return " + ".join(parts) if parts else "（未找到 PDF）"


def pick_examples(rows: list[dict], mode: str, limit: int = 4) -> list[str]:
    """挑出指定模式下页码未命中的题号，用来举例说明检索失败。"""
    return [r["id"] for r in rows
            if not r["refuse"]
            and (r[mode].get("error") is None)
            and not r[mode]["score"]["page_hit"]][:limit]


def make_findings(rows: list[dict], base_sum: dict, agent_sum: dict) -> list[str]:
    """根据本次结果生成结论。不写死任何数字，换策略重跑后报告会自动更新。"""
    valid = [r for r in rows
             if r["baseline"].get("error") is None and r["agent"].get("error") is None]
    b_ok = sum(1 for r in valid if r["baseline"]["score"]["passed"])
    a_ok = sum(1 for r in valid if r["agent"]["score"]["passed"])
    findings = []

    # 1) 通过率与分歧题
    diverge = [r for r in valid if r["baseline"]["score"]["passed"] != r["agent"]["score"]["passed"]]
    if not diverge:
        findings.append(
            f"**两者通过率相同**（各 {b_ok}/{len(valid)}）。在当前语料上 Agent 没有拿到额外通过条数，"
            "收益体现在答案完整度与过程可控性，而不是“多做对几题”。"
        )
    else:
        only_agent = [r["id"] for r in diverge if r["agent"]["score"]["passed"]]
        only_base = [r["id"] for r in diverge if r["baseline"]["score"]["passed"]]
        detail = []
        if only_agent:
            detail.append(f"仅 Agent 通过：{', '.join(only_agent)}")
        if only_base:
            detail.append(f"仅 baseline 通过：{', '.join(only_base)}")
        findings.append(
            f"**通过率 {b_ok}/{len(valid)} vs {a_ok}/{len(valid)}**（baseline vs Agent）。"
            + "；".join(detail) + "。"
        )

    # 2) token 代价
    ratio = agent_sum["平均输入token"] / (base_sum["平均输入token"] or 1)
    findings.append(
        f"**Agent 的输入 token 是 baseline 的 {ratio:.1f} 倍**"
        f"（{base_sum['平均输入token']:.0f} → {agent_sum['平均输入token']:.0f}），"
        f"延迟中位数 {base_sum['延迟中位数']:.2f}s → {agent_sum['延迟中位数']:.2f}s。"
        "代价来自每轮重发完整历史、多轮检索与整页精读；这是 Agent 固有的成本结构，不是实现缺陷。"
    )
    # 平均值与中位数差距悬殊时，说明有长尾请求，要说清楚，否则读者会以为延迟真这么高
    if base_sum["平均延迟"] > base_sum["延迟中位数"] * 3:
        findings[-1] += (
            f"（baseline 平均延迟 {base_sum['平均延迟']:.1f}s 被个别超时重试拉高，不代表常态）"
        )

    # 3) 跨文档题：数据扩充后新增的观察维度
    cross = [r for r in valid if r.get("cross_doc")]
    if cross:
        cb = sum(1 for r in cross if r["baseline"]["score"]["passed"])
        ca = sum(1 for r in cross if r["agent"]["score"]["passed"])
        findings.append(
            f"**跨文档题 baseline {cb}/{len(cross)} vs Agent {ca}/{len(cross)}**。"
            "这类题要求同时取到两份年报的数据再合并，固定管线只做一次 top-k 检索，"
            "更容易只拿到其中一份文档。"
        )

    # 4) 检索命中情况：这次语料变大后最值得看的指标
    b_miss = pick_examples(rows, "baseline")
    a_miss = pick_examples(rows, "agent")
    b_rate = base_sum["页码命中率"]
    # 结论跟着数据走：固定管线已接近全覆盖时，再说“检索掣肘”就是自相矛盾
    tail = (
        "固定管线的单次 top-k 已能覆盖绝大多数目标页，Agent 的增量价值转到了"
        "多轮追问、口径核对与跨文档合并上。"
        if b_rate >= 0.85 else
        f"固定管线仍有约 {(1 - b_rate) * 100:.0f}% 的目标页落在 top-k 之外，"
        "Agent 靠多轮检索把它补了回来。"
    )
    findings.append(
        f"**页码命中率 baseline {pct(b_rate)} / Agent {pct(agent_sum['页码命中率'])}**。"
        + (
            f"baseline 未命中预期页的题：{', '.join(b_miss)}" + (f"；Agent 未命中：{', '.join(a_miss)}" if a_miss else "，Agent 全部命中") + "。"
            if b_miss or a_miss else "两者都能取到正确页面。"
        )
        + tail
    )

    # 5) 拒答
    refused = [r for r in valid if r["refuse"]]
    if refused:
        rb = sum(1 for r in refused if r["baseline"]["score"]["passed"])
        ra = sum(1 for r in refused if r["agent"]["score"]["passed"])
        findings.append(
            f"**拒答正确率 baseline {rb}/{len(refused)} / Agent {ra}/{len(refused)}**。"
            "拒答题考的是“不硬编”：文档里没有的数据，能不能明确说找不到而不是编一个合理数字。"
        )
    return findings


def render_report(rows, base_sum, agent_sum, elapsed) -> str:
    cats = []
    for r in rows:
        if r["category"] not in cats:
            cats.append(r["category"])

    lines = [
        "# 评测报告：baseline vs Agent",
        "",
        f"- 语料：{describe_corpus()}",
        f"- 题目：{len(rows)} 条（单跳 / 多跳 / 跨页对比 / 拒答）",
        f"- 入库：chunk_size={CHUNK_SIZE} / overlap={CHUNK_OVERLAP}，"
        f"丢弃 <{MIN_PAGE_CHARS} 字的页与 <{MIN_CHUNK_CHARS} 字的块；检索 top_k={settings.retrieval_top_k}",
        f"- 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}，模型调用累计耗时 {elapsed:.1f}s",
        "- 判分方式：要点关键词覆盖（≥60%）+（文件, 页码）命中；拒答题命中否定表述即通过",
        "- 检索层指标（不经模型、只看向量召回）见 `evaluation/retrieval.md`",
        "",
        "## 汇总对比",
        "",
        "| 指标 | Baseline（固定管线） | Agent | 变化 |",
        "|---|---|---|---|",
    ]

    def cell(key, value):
        if isinstance(value, str):
            return value
        return pct(value) if key.endswith("率") else f"{value:.2f}"

    def delta(key):
        b, a = base_sum[key], agent_sum[key]
        if isinstance(b, str):
            # "17/18" 这类分数型指标，按通过条数求差
            try:
                b_ok = int(b.split("/")[0])
                a_ok = int(a.split("/")[0])
                return f"{a_ok - b_ok:+d} 条"
            except ValueError:
                return "—"
        if key.endswith("率"):
            return f"{(a - b) * 100:+.1f} pp"
        return f"{a - b:+.2f}"

    for key in ["通过率", "要点覆盖率", "页码命中率", "拒答正确率",
                "平均步数", "平均工具调用", "延迟中位数", "平均延迟",
                "平均输入token", "平均输出token"]:
        b, a = base_sum[key], agent_sum[key]
        lines.append(f"| {key} | {cell(key, b)} | {cell(key, a)} | {delta(key)} |")

    findings = make_findings(rows, base_sum, agent_sum)
    lines += ["", "## 关键结论", ""]
    lines += [f"{i}. {text}" for i, text in enumerate(findings, 1)]
    lines += [
        "",
        "## 评测局限",
        "",
        "- **判分靠关键词**：中文表述变体多，已用基础否定词兜底拒答题，仍可能误判；"
        "宁可接受少量误判，也不引入 LLM-as-judge 的二次不确定性。",
        f"- **样本量 {len(rows)} 条**：足以发现系统性问题，但不足以做统计显著性结论。",
        "- **只统计命中，不算排序质量**：`retrieval.md` 补了检索层的页码命中与要点覆盖，"
        "但仍没有 MRR / nDCG；做 Rerank 时需要补上，否则分不清“排得更前”与“只是碰巧召回”。",
        "",
    ]

    lines += ["", "## 分类通过率", "", "| 类别 | 题数 | Baseline | Agent |", "|---|---|---|---|"]
    for cat in cats:
        subset = [r for r in rows if r["category"] == cat]
        b_ok = sum(1 for r in subset if r["baseline"].get("score", {}).get("passed"))
        a_ok = sum(1 for r in subset if r["agent"].get("score", {}).get("passed"))
        lines.append(f"| {cat} | {len(subset)} | {b_ok}/{len(subset)} | {a_ok}/{len(subset)} |")

    # 跨文档题单独统计：这是多文档语料才有的能力维度
    cross = [r for r in rows if r.get("cross_doc")]
    if cross:
        cb = sum(1 for r in cross if r["baseline"].get("score", {}).get("passed"))
        ca = sum(1 for r in cross if r["agent"].get("score", {}).get("passed"))
        lines.append(f"| （其中）跨文档题 | {len(cross)} | {cb}/{len(cross)} | {ca}/{len(cross)} |")

    lines += [
        "",
        "## 逐题明细",
        "",
        "| ID | 类别 | 问题 | 预期页 | Baseline 要点/页码 | Agent 要点/页码 | Agent 步数 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        b, a = r["baseline"], r["agent"]
        b_cell = "ERR" if b.get("error") else f"{b['score']['point_rate']:.2f} / {'命中' if b['score']['page_hit'] else '未命中'}"
        a_cell = "ERR" if a.get("error") else f"{a['score']['point_rate']:.2f} / {'命中' if a['score']['page_hit'] else '未命中'}"
        if r["refuse"]:
            b_cell = "ERR" if b.get("error") else ("正确拒答" if b["score"]["passed"] else "未拒答")
            a_cell = "ERR" if a.get("error") else ("正确拒答" if a["score"]["passed"] else "未拒答")
        q = r["question"].replace("|", "／")
        pages = ",".join(str(p) for p in r["expect_pages"]) or "—"
        lines.append(f"| {r['id']} | {r['category']} | {q} | {pages} | {b_cell} | {a_cell} | {a.get('steps', 0)} |")

    return "\n".join(lines) + "\n"


def rescore_existing() -> None:
    """不调 API，用现有 results.json 的答案重新判分。

    改关键词表后不必重跑全量，既省钱也保证两次报告用的是同一批答案。
    """
    data_path = PROJECT_ROOT / "evaluation" / "results.json"
    rows = json.loads(data_path.read_text(encoding="utf-8"))
    by_id = {q["id"]: q for q in QUESTIONS}

    for row in rows:
        item = by_id[row["id"]]
        # 同步题库元数据：只改分数而留着旧页码，报告会向导出的预期页自相矛盾
        row["expect_pages"] = item["expect_pages"]
        row["expect_files"] = item.get("expect_files", [])
        row["cross_doc"] = len(item.get("expect_files") or []) > 1
        for mode in ("baseline", "agent"):
            cell = row.get(mode) or {}
            if cell.get("error") is None:
                cell["score"] = score(cell["answer"], cell["sources"], item)

    data_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    elapsed = sum(
        r.get("agent", {}).get("latency", 0) + r.get("baseline", {}).get("latency", 0)
        for r in rows
    )
    report = render_report(
        rows, build_summary(rows, "baseline"), build_summary(rows, "agent"), elapsed
    )
    out = PROJECT_ROOT / "evaluation" / "report.md"
    out.write_text(report, encoding="utf-8")
    print(f"已离线重评：{out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="按类别轮流取前 N 条，0 表示全部")
    parser.add_argument("--mode", choices=["both", "baseline", "agent"], default="both")
    parser.add_argument("--rescore", action="store_true", help="不调 API，仅用已有结果重新判分")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "evaluation" / "report.md"))
    args = parser.parse_args()

    if args.rescore:
        rescore_existing()
        return

    questions = pick_round_robin(QUESTIONS, args.limit)
    db = index_documents()
    agent, collected = build_agent(db) if args.mode in {"both", "agent"} else (None, None)
    llm = get_llm(settings.llm_temperature) if args.mode in {"both", "baseline"} else None

    rows, started = [], time.time()
    for item in questions:
        row = {
            "id": item["id"], "category": item["category"], "question": item["question"],
            "expect_pages": item["expect_pages"], "refuse": item["refuse"],
            "expect_files": item.get("expect_files", []),
            "cross_doc": len(item.get("expect_files") or []) > 1,
        }
        for mode in ("baseline", "agent"):
            if args.mode not in {"both", mode}:
                continue
            try:
                if mode == "baseline":
                    retriever = db.as_retriever(search_kwargs={"k": settings.retrieval_top_k})
                    docs = retriever.invoke(item["question"])
                    start = time.perf_counter()
                    msg = (RAG_PROMPT | llm).invoke({
                        "context": format_docs(docs), "question": item["question"],
                    })
                    latency = time.perf_counter() - start
                    usage = getattr(msg, "usage_metadata", None) or {}
                    result = {
                        "answer": msg.content, "sources": docs_to_sources(docs),
                        "steps": 1, "tool_calls": 1, "latency": latency,
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                    }
                else:
                    result = run_agent(agent, collected, item["question"], f"eval-{item['id']}")
                result["score"] = score(result["answer"], result["sources"], item)
            except Exception as exc:
                result = {"answer": "", "sources": [], "steps": 0, "tool_calls": 0,
                          "latency": 0.0, "input_tokens": 0, "output_tokens": 0,
                          "error": f"{type(exc).__name__}: {exc}",
                          "score": {"point_rate": 0.0, "page_hit": False, "passed": False}}
            row[mode] = result
            flag = "OK " if result["score"]["passed"] else "FAIL"
            print(f"[{flag}] {item['id']:<3} {mode:<8} {result['latency']:>5.1f}s  {item['question'][:34]}")

        rows.append(row)

    elapsed = time.time() - started
    Path(args.out).write_text(
        render_report(rows, build_summary(rows, "baseline"), build_summary(rows, "agent"), elapsed),
        encoding="utf-8",
    )
    Path(PROJECT_ROOT / "evaluation" / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n报告已生成：{args.out}")
    print(f"原始结果：{PROJECT_ROOT / 'evaluation' / 'results.json'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
