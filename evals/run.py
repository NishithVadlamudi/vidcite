"""Run the eval set against the live database:

    python -m evals.run                       # retrieval + confidence gate (no LLM calls)
    python -m evals.run --llm                 # also full answers via the LLM (slow, rate-limited)
    python -m evals.run --source whisper      # compare transcript sources

The MIN_SCORE cutoff is tuned on the dev split and reported on the holdout split,
so the holdout number is an honest estimate. Raw results go to data/evals/.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from evals.metrics import (TOLERANCE_S, best_threshold, first_hit_rank, gate_scores,
                           link_error, summarize_retrieval)
from ingest.chunk import format_ts

QUESTIONS = Path(__file__).with_name("questions.jsonl")
OUT_DIR = Path("data/evals")
THRESHOLDS = [round(0.40 + 0.01 * i, 2) for i in range(41)]  # 0.40 .. 0.80


def load_questions() -> list[dict]:
    return [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]


def run_retrieval(conn, questions, source: str, k: int) -> list[dict]:
    from retrieval.search import retrieve

    rows = []
    for q in questions:
        passages, _ = retrieve(conn, q["question"], k=k, source=source, min_score=-1.0)
        row = {**q, "answerable": q["video_id"] is not None,
               "top_score": passages[0].score if passages else 0.0,
               "top": [(p.video_id, p.start_s, p.end_s, round(p.score, 3)) for p in passages]}
        if row["answerable"]:
            rank = first_hit_rank(passages, q["video_id"], q["gold_s"])
            row["rank"] = rank
            row["link_err"] = link_error(passages[rank - 1].start_s, q["gold_s"]) if rank else None
        rows.append(row)
    return rows


def run_llm(conn, rows, source: str, k: int, delay_s: float) -> None:
    """Adds the full answer to each row. Sleeps between calls for free-tier rate limits."""
    from api.answer import answer_question
    from api.llm import LLMUnavailable

    for i, row in enumerate(rows):
        if i:
            time.sleep(delay_s)
        try:
            r = answer_question(conn, row["question"], source=source, k=k)
        except LLMUnavailable as e:
            row["llm"] = {"error": str(e)}
            continue
        cited_ok = [c for c in r["citations"]
                    if row["answerable"] and c["video_id"] == row["video_id"]
                    and c["start_s"] <= row["gold_s"] <= c["end_s"]]
        row["llm"] = {"found": r["found"], "answer": r["answer"], "citations": r["citations"],
                      "cited_gold": bool(cited_ok), "latency_ms": r["latency_ms"],
                      "link_err": link_error(cited_ok[0]["start_s"], row["gold_s"]) if cited_ok else None}


def pct(x) -> str:
    return "  n/a" if x is None else f"{100 * x:4.0f}%"


def report(rows, k: int, current_min: float) -> dict:
    out = {}
    for split in ("dev", "holdout", "all"):
        part = [r for r in rows if split == "all" or r["split"] == split]
        out[split] = summarize_retrieval([r for r in part if r["answerable"]], k)
    print(f"\nRetrieval (answerable questions, k={k})")
    print(f"  {'split':8} {'n':>3} {'hit@1':>6} {f'hit@{k}':>6} {'MRR':>6} {'link<=15s':>10} {'link err med/max':>17}")
    for split, s in out.items():
        print(f"  {split:8} {s['n']:>3} {pct(s['hit@1']):>6} {pct(s[f'hit@{k}']):>6} {s['mrr']:>6.2f} "
              f"{pct(s['link_on_time']):>10} {s['link_err_median_s']:>8.0f}s/{s['link_err_max_s']:>4.0f}s")

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
    out["gate"] = {"current": current_min, "tuned": t}

    misses = [r for r in rows if r["answerable"] and r["rank"] != 1]
    if misses:
        print("\nNot ranked first:")
        for r in misses:
            got = r["top"][0]
            print(f"  rank={r['rank']}  gold {r['video_id']}@{format_ts(r['gold_s'])}  "
                  f"top1 {got[0]}@{format_ts(got[1])}-{format_ts(got[2])} ({got[3]})  {r['question']}")
    late = [r for r in rows if r["answerable"] and r["link_err"] is not None and r["link_err"] > TOLERANCE_S]
    if late:
        print(f"\nLinks more than {TOLERANCE_S:.0f}s early: {len(late)}/{sum(r['answerable'] for r in rows)}")
    return out


def report_llm(rows) -> dict:
    ans = [r for r in rows if r["answerable"] and "error" not in r["llm"]]
    una = [r for r in rows if not r["answerable"] and "error" not in r["llm"]]
    errors = sum("error" in r["llm"] for r in rows)
    errs = [r["llm"]["link_err"] for r in ans if r["llm"]["link_err"] is not None]
    lat = sorted(r["llm"]["latency_ms"] for r in ans + una)
    s = {
        "answered": sum(r["llm"]["found"] for r in ans) / len(ans) if ans else None,
        "cited_gold": sum(r["llm"]["cited_gold"] for r in ans) / len(ans) if ans else None,
        "abstained": sum(not r["llm"]["found"] for r in una) / len(una) if una else None,
        "link_on_time": sum(e <= TOLERANCE_S for e in errs) / len(errs) if errs else None,
        "latency_p50_ms": lat[len(lat) // 2] if lat else None,
        "errors": errors,
    }
    print("\nEnd to end (LLM)")
    print(f"  answerable answered {pct(s['answered'])}   cited the gold moment {pct(s['cited_gold'])}   "
          f"cited link <=15s early {pct(s['link_on_time'])}")
    print(f"  unanswerable abstained {pct(s['abstained'])}   p50 latency {s['latency_p50_ms']} ms   LLM errors {errors}")
    return s


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate retrieval and answers against evals/questions.jsonl")
    p.add_argument("--source", default="youtube", choices=["youtube", "whisper"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--llm", action="store_true", help="also generate answers (uses LLM quota)")
    p.add_argument("--llm-delay", type=float, default=7.0, help="seconds between LLM calls")
    a = p.parse_args()

    from ingest.store import connect
    from retrieval.search import MIN_SCORE

    conn = connect()
    try:
        rows = run_retrieval(conn, load_questions(), a.source, a.k)
        summary = report(rows, a.k, MIN_SCORE)
        if a.llm:
            run_llm(conn, rows, a.source, a.k, a.llm_delay)
            summary["llm"] = report_llm(rows)
    finally:
        conn.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{a.source}.json"
    out.write_text(json.dumps({"args": vars(a), "summary": summary, "rows": rows}, indent=1, default=str))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
