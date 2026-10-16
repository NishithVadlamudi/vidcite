"""Eval metrics as pure functions, so they can be tested without a database.

A passage is correct when it comes from the gold video and its time span contains
the gold second (the moment the answer is spoken). Separately, a citation link is
"on time" when it starts at most TOLERANCE_S before that moment (links jump to the
passage start, so they can only be early, never late).
"""
from __future__ import annotations

from statistics import median

TOLERANCE_S = 15.0  # the timestamp accuracy target from ingest/captions.py


def contains(passage, video_id: str, gold_s: float) -> bool:
    return passage.video_id == video_id and passage.start_s <= gold_s <= passage.end_s


def first_hit_rank(passages, video_id: str, gold_s: float) -> int | None:
    """1-based rank of the first passage containing the gold moment, or None."""
    for rank, p in enumerate(passages, start=1):
        if contains(p, video_id, gold_s):
            return rank
    return None


def link_error(start_s: float, gold_s: float) -> float:
    """Seconds between where the link jumps to and where the answer is spoken."""
    return gold_s - start_s


def summarize_retrieval(rows: list[dict], k: int) -> dict:
    """rows: answerable questions only, each {"rank": int|None, "link_err": float|None}."""
    n = len(rows)
    if not n:
        return {}
    errs = [r["link_err"] for r in rows if r["link_err"] is not None]
    return {
        "n": n,
        "hit@1": sum(r["rank"] == 1 for r in rows) / n,
        f"hit@{k}": sum(r["rank"] is not None for r in rows) / n,
        "mrr": sum(1 / r["rank"] for r in rows if r["rank"]) / n,
        "link_on_time": sum(e <= TOLERANCE_S for e in errs) / len(errs) if errs else None,
        "link_err_median_s": median(errs) if errs else None,
        "link_err_max_s": max(errs) if errs else None,
    }


def gate_scores(rows: list[dict], threshold: float) -> dict:
    """How well a top_score cutoff separates answerable from unanswerable questions.
    rows: {"answerable": bool, "top_score": float}."""
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    pass_rate = sum(r["top_score"] >= threshold for r in ans) / len(ans) if ans else None
    reject_rate = sum(r["top_score"] < threshold for r in una) / len(una) if una else None
    parts = [x for x in (pass_rate, reject_rate) if x is not None]
    return {"threshold": threshold, "pass_rate": pass_rate, "reject_rate": reject_rate,
            "balanced": sum(parts) / len(parts) if parts else None}


def best_threshold(rows: list[dict], candidates: list[float]) -> dict:
    """Best balanced gate accuracy; ties go to the middle of the tied range, which is
    the least likely to overfit a small dev set."""
    scored = [gate_scores(rows, t) for t in candidates]
    top = max(s["balanced"] for s in scored)
    tied = [s for s in scored if s["balanced"] == top]
    return tied[len(tied) // 2]
