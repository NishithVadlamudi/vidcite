"""Rerank retrieval candidates with a cross-encoder.

Vector search compares two separately computed embeddings; a cross-encoder reads
the question and a chunk together, which is much better at telling "the moment
that answers this" from "a chunk on the same topic" (e.g. a talk's wrap-up that
restates everything). It's too slow to score the whole library, so it only
reorders the top candidates from vector + keyword search.
"""
from __future__ import annotations

import logging
import math
import os

RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = 20  # gold chunk was in the fused top 20 for 52/53 eval questions

_models: dict[str, object] = {}
log = logging.getLogger("rerank")


def get_reranker(name: str | None = None):
    name = name or RERANK_MODEL
    if name not in _models:
        from sentence_transformers import CrossEncoder

        # CPU: one question x 20 chunks is small, same reasoning as query embedding.
        try:  # cached copy, no online check
            _models[name] = CrossEncoder(name, device="cpu", max_length=512, local_files_only=True)
        except OSError:  # not downloaded yet
            _models[name] = CrossEncoder(name, device="cpu", max_length=512)
        # Copy weights out of the memory-mapped safetensors file. With torch 2.14 /
        # transformers 4.57 on Apple Silicon, ms-marco-MiniLM-L-6-v2 returned NaN for every
        # pair when run on the mapped weights, and correct scores after this copy.
        for p in _models[name].model.parameters():
            p.data = p.data.clone()
    return _models[name]


def rerank(question: str, passages: list, k: int, score_pairs=None) -> list:
    """Top k passages by cross-encoder relevance. Passages keep their cosine `score`
    (the confidence gate still uses it); only the order changes."""
    if not passages:
        return []
    if score_pairs is None:
        score_pairs = get_reranker().predict
    scores = [float(s) for s in score_pairs([(question, p.text) for p in passages])]
    if not all(math.isfinite(s) for s in scores):
        # Sorting by NaN silently keeps the input order and looks like a working rerank.
        log.warning("reranker returned non-finite scores; keeping retrieval order")
        return passages[:k]
    order = sorted(range(len(passages)), key=lambda i: -scores[i])  # stable on ties
    return [passages[i] for i in order[:k]]
