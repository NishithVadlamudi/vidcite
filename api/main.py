"""HTTP API and web UI.  Run:  uvicorn api.main:app --reload
then open http://localhost:8000 (UI) or http://localhost:8000/docs (API)."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field

from api.answer import answer_question, search_only
from api.llm import LLMUnavailable

load_dotenv()
pool: ConnectionPool | None = None
log = logging.getLogger("api")
STATIC = Path(__file__).with_name("static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool
    from ingest.store import configure
    pool = ConnectionPool(os.environ["DATABASE_URL"], min_size=1, max_size=4, configure=configure)
    from ingest.embed import QUERY_DEVICE, get_model
    get_model(QUERY_DEVICE)  # load the query embedding model once at startup, not on the first question
    from retrieval.search import RERANK
    if RERANK:
        from retrieval.rerank import get_reranker
        get_reranker()  # same for the cross-encoder (~6 s)
    yield
    pool.close()


app = FastAPI(title="Video Q&A", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    video_id: str | None = None
    source: str = Field(default="auto", pattern="^(auto|youtube|whisper)$")  # auto: Whisper if available


class Citation(BaseModel):
    n: int
    video_id: str
    title: str
    start_s: float
    end_s: float
    link_s: float  # where the cited claim is spoken; use this for the ?t= link


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    found: bool
    top_score: float | None
    latency_ms: int
    search_only: bool = False  # LLM unavailable: citations are the top matching moments, no answer


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):  # sync def: FastAPI runs it in a thread pool
    with pool.connection() as conn:
        try:
            return answer_question(conn, req.question, source=req.source, video_id=req.video_id)
        except LLMUnavailable as e:  # e.g. Gemini free-tier quota or 503: still show the moments
            log.warning("LLM unavailable, returning search-only results: %s", e)
            return search_only(conn, req.question, source=req.source, video_id=req.video_id)


@app.get("/videos")
def videos():
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("select id, title, channel, duration_s, license from videos order by title")
        cols = ["id", "title", "channel", "duration_s", "license"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"ok": True}
