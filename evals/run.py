"""Run the eval set against the live database:

    python -m evals.run                       # retrieval + confidence gate (no LLM calls)
    python -m evals.run --llm                 # also full answers via the LLM (slow, rate-limited)
    python -m evals.run --source youtube      # compare transcript sources (default: auto)
    python -m evals.run --retrieval hybrid    # vector + keyword search fused with RRF
    python -m evals.run --merge on            # join adjacent retrieved chunks
    python -m evals.run --rerank on           # cross-encoder reranking (model: RERANK_MODEL env var)

The MIN_SCORE cutoff is tuned on the dev split and reported on the holdout split,
so the holdout number is an honest estimate. Raw results go to data/evals/.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from evals.metrics import (best_threshold, first_hit, gate_scores, gold_moments, link_error, link_stats,
                           matched_gold, nearest_gold,
                           on_time, summarize_retrieval)
from ingest.chunk import format_ts

QUESTIONS = Path(__file__).with_name("questions.jsonl")
OUT_DIR = Path("data/evals")
THRESHOLDS = [round(0.40 + 0.01 * i, 2) for i in range(41)]  # 0.40 .. 0.80


def load_questions() -> list[dict]:
    return [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]


def run_retrieval(conn, questions, source: str, k: int, hybrid: bool, merge: bool, rerank: bool) -> list[dict]:
    from ingest.embed import QUERY_DEVICE, embed_passages, embed_query, get_model
    from retrieval.moment import best_moment
    from retrieval.search import retrieve, top_score

    embed_texts = lambda texts: embed_passages(texts, model=get_model(QUERY_DEVICE))
    rows = []
    for q in questions:
        passages, _ = retrieve(conn, q["question"], k=k, source=source, min_score=-1.0,
                               hybrid=hybrid, merge=merge, rerank=rerank)
        row = {**q, "answerable": q["video_id"] is not None,
               "top_score": top_score(passages) if passages else 0.0,
               "keyword_pass": any(p.all_terms for p in passages),
               "n_passages": len(passages),
               "context_s": sum(p.end_s - p.start_s for p in passages),
               "top": [(p.video_id, p.start_s, p.end_s, round(p.score, 3)) for p in passages]}
        if row["answerable"]:
            golds = gold_moments(q)
            rank, _ = first_hit(passages, golds)
            row["rank"] = rank
            row["link_err"] = row["moment_err"] = None
            if rank:
                hit = passages[rank - 1]
                row["link_err"] = link_error(hit.start_s, nearest_gold(hit, golds, hit.start_s))
                # Sentence-level link, using the question as the claim (no LLM needed).
                moment = best_moment(hit, embed_query(q["question"]), embed_texts)
                row["moment_err"] = link_error(moment, nearest_gold(hit, golds, moment))
        rows.append(row)
    return rows


class _Span:
    """Lets a citation dict be checked with the same `contains` as a passage."""
    def __init__(self, c: dict):
        self.video_id, self.start_s, self.end_s = c["video_id"], c["start_s"], c["end_s"]


def run_llm(conn, rows, source: str, k: int, delay_s: float, hybrid: bool, merge: bool,
            rerank: bool = False) -> None:
    """Adds the full answer to each row. Sleeps between calls for free-tier rate limits."""
    from api.answer import answer_question
    from api.llm import LLMUnavailable

    for i, row in enumerate(rows):
        if i:
            time.sleep(delay_s)
        try:
            r = answer_question(conn, row["question"], source=source, k=k, hybrid=hybrid, merge=merge, rerank=rerank)
        except LLMUnavailable as e:
            row["llm"] = {"error": str(e)}
            continue
        golds = gold_moments(row)
        cited_ok = [c for c in r["citations"] if matched_gold(_Span(c), golds) is not None]
        first = cited_ok[0] if cited_ok else None
        row["llm"] = {"found": r["found"], "answer": r["answer"], "citations": r["citations"],
                      "cited_gold": bool(cited_ok), "latency_ms": r["latency_ms"],
                      "chunk_link_err": link_error(first["start_s"], nearest_gold(_Span(first), golds, first["start_s"])) if first else None,
                      "link_err": link_error(first["link_s"], nearest_gold(_Span(first), golds, first["link_s"])) if first else None}


def pct(x) -> str:
    return "  n/a" if x is None else f"{100 * x:4.0f}%"


def report(rows, k: int, current_min: float) -> dict:
    out = {}
    for split in ("dev", "holdout", "all"):
        part = [r for r in rows if split == "all" or r["split"] == split]
        out[split] = summarize_retrieval([r for r in part if r["answerable"]], k)
    print(f"\nRetrieval (answerable questions, k={k}). Links on time = 3 s late to 15 s early.")
    print(f"  {'split':8} {'n':>3} {'hit@1':>6} {f'hit@{k}':>6} {'MRR':>6}   {'chunk-start link':>22}   {'sentence link':>22}")
    for split, s in out.items():
        print(f"  {split:8} {s['n']:>3} {pct(s['hit@1']):>6} {pct(s[f'hit@{k}']):>6} {s['mrr']:>6.2f}   "
              f"{pct(s['link_on_time'])} on time, {s['link_abs_median_s']:>3.0f}s med   "
              f"{pct(s['moment_on_time'])} on time, {s['moment_abs_median_s']:>3.0f}s med")

    dev = [r for r in rows if r["split"] == "dev"]
    hold = [r for r in rows if r["split"] == "holdout"]
    best = best_threshold(dev, THRESHOLDS)
    t = best["threshold"]
    print(f"\nConfidence gate (answerable should pass, unanswerable should be rejected)")
    print(f"  {'cutoff':22} {'split':8} {'pass':>6} {'reject':>7} {'balanced':>9}")
    for label, thr in ((f"current MIN_SCORE {current_min:.2f}", current_min), (f"tuned on dev   {t:.2f}", t)):
        for split, part in (("dev", dev), ("holdout", hold)):
            g = gate_scores(part, thr)
            print(f"  {label:22} {split:8} {pct(g['pass_rate']):>6} {pct(g['reject_rate']):>7} {pct(g['balanced']):>9}")
    ans = sorted(r["top_score"] for r in rows if r["answerable"])
    una = sorted(r["top_score"] for r in rows if not r["answerable"])
    print(f"  top_score ranges: answerable {ans[0]:.3f}-{ans[-1]:.3f}, unanswerable {una[0]:.3f}-{una[-1]:.3f}")
    kw = [r for r in rows if r.get("keyword_pass")]
    print(f"  keyword pass (every question word in one chunk): "
          f"{sum(r['answerable'] for r in kw)} answerable, {sum(not r['answerable'] for r in kw)} unanswerable")
    print(f"  passages per question {sum(r['n_passages'] for r in rows) / len(rows):.1f}, "
          f"context {sum(r['context_s'] for r in rows) / len(rows):.0f} s of talk")
    out["gate"] = {"current": current_min, "tuned": t}

    misses = [r for r in rows if r["answerable"] and r["rank"] != 1]
    if misses:
        print("\nNot ranked first:")
        for r in misses:
            got = r["top"][0]
            print(f"  rank={r['rank']}  gold {r['video_id']}@{format_ts(r['gold_s'])}  "
                  f"top1 {got[0]}@{format_ts(got[1])}-{format_ts(got[2])} ({got[3]})  {r['question']}")
    off = [r for r in rows if r["answerable"] and r["moment_err"] is not None and not on_time(r["moment_err"])]
    if off:
        print("\nSentence links off target (+ early / - late):")
        for r in off:
            print(f"  {r['moment_err']:+5.0f}s  gold {format_ts(r['gold_s'])}  {r['question']}")
    return out


def report_llm(rows) -> dict:
    ans = [r for r in rows if r["answerable"] and "error" not in r["llm"]]
    una = [r for r in rows if not r["answerable"] and "error" not in r["llm"]]
    errors = sum("error" in r["llm"] for r in rows)
    lat = sorted(r["llm"]["latency_ms"] for r in ans + una)
    s = {
        "answered": sum(r["llm"]["found"] for r in ans) / len(ans) if ans else None,
        "cited_gold": sum(r["llm"]["cited_gold"] for r in ans) / len(ans) if ans else None,
        "abstained": sum(not r["llm"]["found"] for r in una) / len(una) if una else None,
        **link_stats([r["llm"]["chunk_link_err"] for r in ans if r["llm"]["chunk_link_err"] is not None], "chunk_link"),
        **link_stats([r["llm"]["link_err"] for r in ans if r["llm"]["link_err"] is not None], "link"),
        "latency_p50_ms": lat[len(lat) // 2] if lat else None,
        "errors": errors,
    }
    print("\nEnd to end (LLM)")
    print(f"  answerable answered {pct(s['answered'])}   cited the gold moment {pct(s['cited_gold'])}")
    print(f"  cited links on time: chunk start {pct(s['chunk_link_on_time'])}, "
          f"sentence {pct(s['link_on_time'])} (median off by {s['link_abs_median_s'] or 0:.1f}s)")
    print(f"  unanswerable abstained {pct(s['abstained'])}   p50 latency {s['latency_p50_ms']} ms   LLM errors {errors}")
    return s


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate retrieval and answers against evals/questions.jsonl")
    p.add_argument("--source", default="auto", choices=["auto", "youtube", "whisper"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--retrieval", default=None, choices=["vector", "hybrid"],
                   help="default: the app's setting (HYBRID env var)")
    p.add_argument("--merge", default=None, choices=["on", "off"], help="default: the app's setting (MERGE env var)")
    p.add_argument("--rerank", default=None, choices=["on", "off"], help="default: the app's setting (RERANK env var)")
    p.add_argument("--llm", action="store_true", help="also generate answers (uses LLM quota)")
    p.add_argument("--llm-delay", type=float, default=7.0, help="seconds between LLM calls")
    a = p.parse_args()

    from ingest.store import connect
    from retrieval.rerank import RERANK_MODEL
    from retrieval.search import HYBRID, MERGE, MIN_SCORE, RERANK

    hybrid = HYBRID if a.retrieval is None else a.retrieval == "hybrid"
    merge = MERGE if a.merge is None else a.merge == "on"
    a.retrieval = "hybrid" if hybrid else "vector"
    a.merge = "on" if merge else "off"
    rerank = RERANK if a.rerank is None else a.rerank == "on"
    a.rerank = RERANK_MODEL.split("/")[-1] if rerank else "off"
    print(f"source={a.source}  retrieval={a.retrieval}  merge={a.merge}  rerank={a.rerank}  k={a.k}")
    conn = connect()
    try:
        t0 = time.perf_counter()
        rows = run_retrieval(conn, load_questions(), a.source, a.k, hybrid, merge, rerank)
        print(f"retrieval for {len(rows)} questions took {time.perf_counter() - t0:.1f}s "
              "(includes model loading and sentence-link embedding)")
        summary = report(rows, a.k, MIN_SCORE)
        if a.llm:
            run_llm(conn, rows, a.source, a.k, a.llm_delay, hybrid, merge, rerank)
            summary["llm"] = report_llm(rows)
    finally:
        conn.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{a.source}-{a.retrieval}-merge{a.merge}-rerank{a.rerank}.json"
    out.write_text(json.dumps({"args": vars(a), "summary": summary, "rows": rows}, indent=1, default=str))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
