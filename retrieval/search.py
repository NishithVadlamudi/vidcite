"""Retrieval: vector search, optional keyword search fused with reciprocal rank
fusion (RRF), optional neighbour merging, and a confidence gate.

Keyword search catches rare exact terms that embeddings blur ("postmaster", "LSN").
Neighbour merging joins retrieved chunks that sit next to each other in a video, so
the LLM reads one stretch of the talk instead of two overlapping fragments (which
also produced duplicate citations). Each step can be switched off so the eval can
measure it (python -m evals.run --retrieval vector|hybrid --merge on|off).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace

from retrieval.moment import SENTENCE_BREAK, sentence_times

# Below this cosine similarity we say "not found" instead of asking the LLM.
# A starting guess; Phase 3 tunes it on the eval set.
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.55"))
# On by default: in the eval it raised hit@1 59% -> 76% and hit@5 93% -> 100%.
HYBRID = os.environ.get("HYBRID", "1") == "1"
MERGE = os.environ.get("MERGE", "0") == "1"
RRF_K = 60          # standard RRF damping constant
CANDIDATES = 20     # per search method, before fusion

VECTOR_SEARCH = """
select c.video_id, v.title, c.chunk_index, c.start_s, c.end_s, c.text,
       1 - (c.embedding <=> %(q)s) as score
from chunks c
join videos v on v.id = c.video_id
where c.source = %(source)s
  and (%(video_id)s::text is null or c.video_id = %(video_id)s)
order by c.embedding <=> %(q)s
limit %(k)s
"""

# Any-word match (plainto_tsquery ANDs the words; we OR them), ranked by cover density.
# Also returns cosine similarity so the confidence gate works the same for both lists,
# and all_terms: whether the chunk contains every content word of the question.
KEYWORD_SEARCH = """
select c.video_id, v.title, c.chunk_index, c.start_s, c.end_s, c.text,
       1 - (c.embedding <=> %(q)s) as score,
       c.tsv @@ plainto_tsquery('english', %(text)s) as all_terms
from chunks c
join videos v on v.id = c.video_id
cross join (select replace(plainto_tsquery('english', %(text)s)::text, '&', '|')::tsquery as query) q
where c.source = %(source)s
  and (%(video_id)s::text is null or c.video_id = %(video_id)s)
  and c.tsv @@ query
order by ts_rank_cd(c.tsv, query) desc
limit %(k)s
"""


@dataclass
class Passage:
    video_id: str
    title: str
    chunk_index: int
    start_s: float
    end_s: float
    text: str
    score: float
    all_terms: bool = False  # keyword search: chunk contains every content word of the question
    times: list | None = None  # merged passages: [(start_s, sentence)] estimated per original chunk


def vector_search(conn, query_vec, k: int = 5, source: str = "youtube", video_id: str | None = None) -> list[Passage]:
    with conn.cursor() as cur:
        cur.execute(VECTOR_SEARCH, {"q": query_vec, "source": source, "video_id": video_id, "k": k})
        return [Passage(*row) for row in cur.fetchall()]


def keyword_search(conn, query_vec, question: str, k: int = 5, source: str = "youtube",
                   video_id: str | None = None) -> list[Passage]:
    with conn.cursor() as cur:
        cur.execute(KEYWORD_SEARCH, {"q": query_vec, "text": question, "source": source,
                                     "video_id": video_id, "k": k})
        return [Passage(*row) for row in cur.fetchall()]


def rrf(ranked_lists: list[list[Passage]], k: int, rrf_k: int = RRF_K) -> list[Passage]:
    """Reciprocal rank fusion: each list adds 1/(rrf_k + rank) to a passage's fused score.
    Passages keep their cosine `score`; only the order comes from fusion."""
    fused: dict[tuple[str, int], float] = {}
    first: dict[tuple[str, int], Passage] = {}
    for passages in ranked_lists:
        for rank, p in enumerate(passages, start=1):
            key = (p.video_id, p.chunk_index)
            fused[key] = fused.get(key, 0.0) + 1 / (rrf_k + rank)
            first.setdefault(key, p)
            if p.all_terms and not first[key].all_terms:
                first[key] = replace(first[key], all_terms=True)
    order = sorted(fused, key=lambda key: -fused[key])  # stable: ties keep first-seen order
    return [first[key] for key in order[:k]]


def merge_neighbours(passages: list[Passage]) -> list[Passage]:
    """Join retrieved chunks that are adjacent in the same video into one passage,
    dropping the overlap sentence they share. Keeps rank order (by best member)."""
    by_key = {(p.video_id, p.chunk_index): p for p in passages}
    used: set[tuple[str, int]] = set()
    merged = []
    for p in passages:
        if (p.video_id, p.chunk_index) in used:
            continue
        lo = hi = p.chunk_index
        while (p.video_id, lo - 1) in by_key and (p.video_id, lo - 1) not in used:
            lo -= 1
        while (p.video_id, hi + 1) in by_key and (p.video_id, hi + 1) not in used:
            hi += 1
        run = [by_key[(p.video_id, i)] for i in range(lo, hi + 1)]
        used.update((p.video_id, i) for i in range(lo, hi + 1))
        merged.append(run[0] if len(run) == 1 else _join(run))
    return merged


def _join(run: list[Passage]) -> Passage:
    # Sentence times are estimated within each original chunk before joining: spreading
    # words evenly over a 90-140 s merged span drifts much more than over one ~45 s chunk.
    times: list[tuple[float, str]] = []
    for p in run:
        nxt = sentence_times(p)
        if times and nxt and nxt[0][1].strip() == times[-1][1].strip():
            nxt = nxt[1:]  # the overlap sentence each chunk repeats from the previous one
        times += nxt
    return Passage(run[0].video_id, run[0].title, run[0].chunk_index, run[0].start_s, run[-1].end_s,
                   " ".join(s for _, s in times), max(p.score for p in run),
                   any(p.all_terms for p in run), times)


def retrieve(conn, question: str, k: int = 5, source: str = "youtube",
             video_id: str | None = None, embed=None, min_score: float = MIN_SCORE,
             hybrid: bool | None = None, merge: bool | None = None) -> tuple[list[Passage], bool]:
    """Returns (passages, found). found=False means the match is too weak to answer from:
    best cosine below min_score and no chunk containing every content word of the question."""
    if embed is None:
        from ingest.embed import embed_query as embed
    if hybrid is None:
        hybrid = HYBRID
    if merge is None:
        merge = MERGE
    import numpy as np

    q = np.asarray(embed(question), dtype=np.float32)
    if hybrid:
        n = max(CANDIDATES, k)
        passages = rrf([vector_search(conn, q, k=n, source=source, video_id=video_id),
                        keyword_search(conn, q, question, k=n, source=source, video_id=video_id)], k=k)
    else:
        passages = vector_search(conn, q, k=k, source=source, video_id=video_id)
    if merge:
        passages = merge_neighbours(passages)
    found = bool(passages) and (top_score(passages) >= min_score or any(p.all_terms for p in passages))
    return passages, found


def top_score(passages: list[Passage]) -> float:
    """Best cosine similarity. With fusion the first passage isn't always the closest."""
    return max(p.score for p in passages)
