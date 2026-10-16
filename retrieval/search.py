"""Retrieval: vector search, optional keyword search fused with reciprocal rank
fusion (RRF), and a confidence gate.

Keyword search catches rare exact terms that embeddings blur ("postmaster", "LSN").
Reranking and neighbour merging are still left out so the eval can measure each
step's gain on its own (python -m evals.run --retrieval vector|hybrid).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Below this cosine similarity we say "not found" instead of asking the LLM.
# A starting guess; Phase 3 tunes it on the eval set.
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.55"))
# On by default: in the eval it raised hit@1 59% -> 76% and hit@5 93% -> 100%.
HYBRID = os.environ.get("HYBRID", "1") == "1"
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
# Also returns cosine similarity so the confidence gate works the same for both lists.
KEYWORD_SEARCH = """
select c.video_id, v.title, c.chunk_index, c.start_s, c.end_s, c.text,
       1 - (c.embedding <=> %(q)s) as score
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
    order = sorted(fused, key=lambda key: -fused[key])  # stable: ties keep first-seen order
    return [first[key] for key in order[:k]]


def retrieve(conn, question: str, k: int = 5, source: str = "youtube",
             video_id: str | None = None, embed=None, min_score: float = MIN_SCORE,
             hybrid: bool | None = None) -> tuple[list[Passage], bool]:
    """Returns (passages, found). found=False means the best match is too weak to answer from."""
    if embed is None:
        from ingest.embed import embed_query as embed
    if hybrid is None:
        hybrid = HYBRID
    import numpy as np

    q = np.asarray(embed(question), dtype=np.float32)
    if hybrid:
        n = max(CANDIDATES, k)
        passages = rrf([vector_search(conn, q, k=n, source=source, video_id=video_id),
                        keyword_search(conn, q, question, k=n, source=source, video_id=video_id)], k=k)
    else:
        passages = vector_search(conn, q, k=k, source=source, video_id=video_id)
    found = bool(passages) and top_score(passages) >= min_score
    return passages, found


def top_score(passages: list[Passage]) -> float:
    """Best cosine similarity. With fusion the first passage isn't always the closest."""
    return max(p.score for p in passages)
