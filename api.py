"""
启动：
    uvicorn api:app --reload --port 8000
    浏览器打开 http://127.0.0.1:8000
"""
import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from agent.graph import build_agent
from config import settings
from rag.chain import astream_answer
from rag.indexer import index_documents
from sources import dedupe_sources, docs_to_sources

WEB_DIR = Path(__file__).resolve().parent / "web"

# 进程内会话记忆：thread_id 由前端生成（页面级 uuid），同一个 id 视为同一会话，
# Agent 自动接续历史。单机演示够用；多实例部署换 SQLite/Redis 版 checkpointer，
# 图本身不用改。并发写同一 session 会串历史，所以前端每次开新会话都换 id。
CHECKPOINTER = MemorySaver()
STATE: dict = {"db": None}

# 会话级引用集合。追问往往不再检索（模型直接拿上一轮的数字计算），但答案仍
# 基于上一轮取到的数，引用列表必须累积——否则答案里的页码在上文找不到出处。
SESSION_SOURCES: dict[str, list[dict]] = {}
# 单个会话最多保留的引用条数，防止长期运行无限增长
MAX_SESSION_SOURCES = 200


@asynccontextmanager
async def lifespan(_: FastAPI):
    # 索引加载与校验是同步阻塞的（内容变更时还会触发 embedding 请求），
    # 丢到线程里跑，避免卡住事件循环导致启动阶段连健康检查都超时
    STATE["db"] = await asyncio.to_thread(index_documents)
    yield
    STATE["db"] = None


app = FastAPI(title="文档研究 Agent", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    mode: str = Field("agent", pattern="^(agent|baseline)$")
    session_id: str | None = Field(None, max_length=64)


def sse(event: str, data: dict) -> str:
    """拼一个 SSE 帧。ensure_ascii=False：中文原样输出，抓包和录屏都看得懂。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def describe_tool(name: str, args: dict) -> str:
    """把工具名 + 参数翻成一句人话，直接显示在步骤卡片上。"""
    if name == "search_knowledge_base":
        return f"检索：{args.get('query', '')}"
    if name == "read_document":
        start = args.get("start_page", 1)
        end = args.get("end_page", start)
        span = f"第 {start} 页" if start == end else f"第 {start}-{end} 页"
        return f"精读 {args.get('file_name', '')} {span}"
    if name == "calculate":
        return f"计算：{args.get('expression', '')}"
    if name == "write_report":
        return f"写报告：{args.get('title', '')}"
    return name


async def stream_baseline(question: str, db):
    """固定管线：先给引用，再逐 token 给答案。"""
    async for kind, payload in astream_answer(question, db):
        if kind == "sources":
            yield sse("sources", {"sources": docs_to_sources(payload)})
        else:
            yield sse("token", {"text": payload})


async def stream_agent(question: str, db, session_id: str, stats: dict):
    """Agent 管线：工具调用即时推送，最终答案逐 token 推送。

    每个请求新建一次图：图的 checkpoint 共用（靠 thread_id 隔离会话），
    但工具闭包引用的是会话级 collected，两者要分开传。
    """
    collected = SESSION_SOURCES.setdefault(session_id, [])
    del collected[:-MAX_SESSION_SOURCES]
    agent, collected = build_agent(db, checkpointer=CHECKPOINTER, collected=collected)
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": settings.agent_max_steps * 2 + 5,
    }
    inputs = {"messages": [HumanMessage(content=question)], "question": question, "steps": 0}
    emitted = []

    async for mode, payload in agent.astream(
        inputs, config=config, stream_mode=["updates", "messages"]
    ):
        if mode == "messages":
            chunk, meta = payload
            # 只有 agent 节点会调模型，tools / guard 节点没有 token 可流
            if meta.get("langgraph_node") != "agent":
                continue
            # 工具调用的参数同样走流式，但它是给程序解析的 JSON，
            # 混进答案会让用户看到一串 {"query": ...}
            if getattr(chunk, "tool_call_chunks", None):
                continue
            text = chunk.content
            if isinstance(text, list):      # 多模态格式下 content 是分块列表
                text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
            if text:
                emitted.append(text)
                yield sse("token", {"text": text})
            continue

        for node, update in (payload or {}).items():
            messages = (update or {}).get("messages") or []
            if node == "agent":
                stats["steps"] = stats.get("steps", 0) + 1
                for message in messages:
                    for call in getattr(message, "tool_calls", None) or []:
                        args = call.get("args") or {}
                        yield sse("tool", {
                            "name": call["name"],
                            "label": describe_tool(call["name"], args),
                            "args": args,
                        })
            elif node == "tools":
                for message in messages:
                    preview = (message.content or "").strip().replace("\n", " ")
                    yield sse("observation", {
                        "name": getattr(message, "name", "") or "",
                        "preview": preview[:160],
                    })

    # 兜底：触发步数守卫时，收尾语是节点里拼的字符串，没有模型调用也就没有
    # token 流。不回补的话前端会得到一片空白。
    if not emitted:
        snapshot = await agent.aget_state(config)
        last = snapshot["values"]["messages"][-1]
        content = getattr(last, "content", "")
        if content:
            yield sse("token", {"text": content})

    sources = dedupe_sources(collected)
    if sources:
        yield sse("sources", {"sources": sources})


@app.post("/api/ask")
async def ask(request: AskRequest):
    db = STATE["db"]
    if db is None:
        raise HTTPException(status_code=503, detail="索引尚未就绪，请稍后重试")
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 没带 session_id 就是一次性问答，给个短 id 即可
    session_id = request.session_id or f"web-{uuid.uuid4().hex[:8]}"
    mode = request.mode

    async def event_stream():
        started = time.perf_counter()
        stats = {"steps": 0}
        yield sse("meta", {"mode": mode, "session_id": session_id})
        try:
            pipeline = (
                stream_baseline(question, db)
                if mode == "baseline"
                else stream_agent(question, db, session_id, stats)
            )
            async for frame in pipeline:
                yield frame
            yield sse("done", {
                "mode": mode,
                "latency": round(time.perf_counter() - started, 2),
                "steps": stats["steps"],
            })
        except asyncio.CancelledError:
            # 客户端主动断开（关页面/切问题）：停止生成，不要把异常当错误推回去
            raise
        except Exception as exc:  # noqa: BLE001 - 任何异常都要变成可读的 SSE 帧
            yield sse("error", {"message": f"{type(exc).__name__}: {exc}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 关掉反向代理的响应缓冲，否则“流式”会被攒成一整块再吐
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/health")
async def health():
    db = STATE["db"]
    index = getattr(db, "index", None)
    return {
        "status": "ok" if db is not None else "loading",
        "chunks": getattr(index, "ntotal", 0),
        "top_k": settings.retrieval_top_k,
        "max_steps": settings.agent_max_steps,
        "llm": settings.deepseek_model,
        "embedding": settings.embedding_model,
    }


@app.get("/")
async def index_page():
    return FileResponse(WEB_DIR / "index.html")
