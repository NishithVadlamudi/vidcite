"""Map the LLM's [S1]-style tags back to stored timestamps.

Timestamps always come from our database rows, never from the model's text.
Tags that point at no passage are removed and reported.
"""
from __future__ import annotations

import re

TAG = re.compile(r"\[S(\d+)\]")


def resolve_citations(answer: str, passages) -> tuple[str, list[dict], list[int]]:
    """Returns (clean_answer, citations in order of first use, invalid tag numbers)."""
    citations: list[dict] = []
    seen: set[int] = set()
    invalid: list[int] = []

    def check(match: re.Match) -> str:
        n = int(match.group(1))
        if not 1 <= n <= len(passages):
            invalid.append(n)
            return ""
        if n not in seen:
            seen.add(n)
            p = passages[n - 1]
            citations.append({
                "n": n, "video_id": p.video_id, "title": p.title,
                "start_s": p.start_s, "end_s": p.end_s,
            })
        return match.group(0)

    clean = TAG.sub(check, answer)
    clean = re.sub(r"[ \t]+([.,;:!?])", r"\1", clean)   # tidy space left by removed tags
    clean = re.sub(r"[ \t]{2,}", " ", clean).strip()
    return clean, citations, invalid
