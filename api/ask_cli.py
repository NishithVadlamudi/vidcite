"""Ask from the terminal, no server needed:
    python -m api.ask_cli "How does Raft elect a leader?" [--source whisper] [--video VIDEO_ID]
"""
import argparse

from api.answer import answer_question
from api.llm import LLMUnavailable
from ingest.chunk import format_ts
from ingest.store import connect


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--source", default="youtube", choices=["youtube", "whisper"])
    p.add_argument("--video")
    a = p.parse_args()

    conn = connect()
    try:
        r = answer_question(conn, a.question, source=a.source, video_id=a.video)
    except LLMUnavailable as e:
        raise SystemExit(f"\nLLM unavailable, try again shortly (or set LLM_MODEL).\n  {e}")
    finally:
        conn.close()
    print("\n" + r["answer"] + "\n")
    # Overlapping chunks can cite the same moment; show each link once with all its tags.
    links: dict[tuple[str, int], list[dict]] = {}
    for c in r["citations"]:
        links.setdefault((c["video_id"], int(c["link_s"])), []).append(c)
    for (video_id, t), cs in links.items():
        tags = "".join(f"[S{c['n']}]" for c in cs)
        print(f"  {tags} {cs[0]['title']} @ {format_ts(t)}  https://youtu.be/{video_id}?t={t}")
    print(f"\n  found={r['found']}  top_score={r['top_score']}  latency={r['latency_ms']} ms")


if __name__ == "__main__":
    main()
