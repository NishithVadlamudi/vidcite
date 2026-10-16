"""Phase 2 retrieval: vector search plus a confidence gate.

Phase 4 adds keyword search, rank fusion, reranking and neighbour merging
here; keeping them out for now lets the eval measure each one's gain.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Below this cosine similarity we say "not found" instead of asking the LLM.
# A starting guess; Phase 3 tunes it on the eval set.
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.55"))

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


def retrieve(conn, question: str, k: int = 5, source: str = "youtube",
             video_id: str | None = None, embed=None, min_score: float = MIN_SCORE) -> tuple[list[Passage], bool]:
    """Returns (passages, found). found=False means the best match is too weak to answer from."""
    if embed is None:
        from ingest.embed import embed_query as embed
    import numpy as np

    q = np.asarray(embed(question), dtype=np.float32)
    passages = vector_search(conn, q, k=k, source=source, video_id=video_id)
    found = bool(passages) and passages[0].score >= min_score
    return passages, found
