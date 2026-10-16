"""Prompt that makes the LLM answer only from numbered passages and tag every
sentence with the passages it used, e.g. "Raft elects a leader [S2]."
"""
from __future__ import annotations

from ingest.chunk import format_ts

NOT_FOUND = "NOT_FOUND"

SYSTEM = f"""You answer questions about technical talks using ONLY the numbered passages provided.

Rules:
- Answer in 2 to 5 sentences, plainly and specifically.
- End every sentence with the tag(s) of the passage(s) that support it, like [S1] or [S2][S4].
- Never use facts that are not in the passages, even if you know them.
- If the passages do not contain the answer, reply with exactly {NOT_FOUND} and nothing else.
- Do not mention "passages" or timestamps in your answer; the tags are enough."""


def build_user_message(question: str, passages) -> str:
    blocks = [
        f"[S{i}] {p.title} ({format_ts(p.start_s)}-{format_ts(p.end_s)})\n{p.text}"
        for i, p in enumerate(passages, start=1)
    ]
    return "Passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
