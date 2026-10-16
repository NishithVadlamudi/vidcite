"""Find the moment inside a chunk where a cited claim is spoken.

Chunks are ~45 s long, so linking to a chunk's start can land well before the
answer. Chunks store only their start and end times, not per-word timings, so a
sentence's start is estimated from its word offset, assuming an even speaking rate
within the chunk. The link goes to the sentence most similar to the claim, minus a
short lead-in: the best-matching sentence is often the explanation, a beat after the
topic is introduced, and estimated times drift. LINK_LEAD_S was swept on the eval's
dev split (0-12 s; 8-10 s tied best) and confirmed on holdout (67% -> 89% on time).
"""
from __future__ import annotations

import re

SENTENCE_BREAK = re.compile(r"(?<=[.?!])\s+")
LINK_LEAD_S = 8.0  # smallest of the tied-best leads: least unrelated speech for the viewer


def sentence_times(passage) -> list[tuple[float, str]]:
    """[(estimated_start_s, sentence), ...] for each sentence in the chunk."""
    sentences = [s for s in SENTENCE_BREAK.split(passage.text) if s.strip()]
    total = sum(len(s.split()) for s in sentences)
    if not total:
        return []
    span = passage.end_s - passage.start_s
    out, seen = [], 0
    for s in sentences:
        out.append((passage.start_s + span * seen / total, s))
        seen += len(s.split())
    return out


def best_moment(passage, query_vec, embed_texts, lead_s: float = LINK_LEAD_S) -> float:
    """Estimated start of the sentence most similar to query_vec (normalized vectors,
    so the dot product is cosine), minus lead_s. Falls back to the chunk start."""
    times = sentence_times(passage)
    if len(times) < 2:
        return passage.start_s
    vectors = embed_texts([s for _, s in times])
    scores = [sum(a * b for a, b in zip(query_vec, v)) for v in vectors]
    best = max(range(len(scores)), key=lambda i: scores[i])  # first one wins ties
    return round(max(0.0, times[best][0] - lead_s), 2)
