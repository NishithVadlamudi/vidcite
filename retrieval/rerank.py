"""Rerank retrieval candidates with a cross-encoder.

Vector search compares two separately computed embeddings; a cross-encoder reads
the question and a chunk together, which is much better at telling "the moment
that answers this" from "a chunk on the same topic" (e.g. a talk's wrap-up that
restates everything). It's too slow to score the whole library, so it only
reorders the top candidates from vector + keyword search.

Chunks are scored by their best 3-sentence window ("MaxP"), not as a whole: the
model was trained on ~60-word passages, and a ~230-word chunk whose answer sits in
the middle of unrelated material scored far below a chunk that is *about* the
topic throughout. On the 65-question eval this took hit@1 42 -> 45/53 and
hit@5 51 -> 52/53 with no losses (window size chosen on dev, confirmed on holdout).
"""
from __future__ import annotations

import logging
import math
import os

RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = 10  # same eval results as 20 candidates, at ~half the reranking cost
WINDOW_SENTENCES = 3

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


def windows(text: str, size: int = WINDOW_SENTENCES) -> list[str]:
    """Overlapping windows of `size` sentences, sliding one sentence at a time.
    Unpunctuated text uses 15-word blocks instead of sentences (as for citation links)."""
    from retrieval.moment import BLOCK_WORDS, SENTENCE_BREAK

    units = [s for s in SENTENCE_BREAK.split(text) if s.strip()]
    if len(units) <= 1:
        words = text.split()
        units = [" ".join(words[i:i + BLOCK_WORDS]) for i in range(0, len(words), BLOCK_WORDS)]
    if len(units) <= size:
        return [" ".join(units)] if units else [text]
    return [" ".join(units[i:i + size]) for i in range(len(units) - size + 1)]


def rerank(question: str, passages: list, k: int, score_pairs=None) -> list:
    """Top k passages by their best window's cross-encoder relevance. Passages keep
    their cosine `score` (the confidence gate still uses it); only the order changes."""
    if not passages:
        return []
    if score_pairs is None:
        score_pairs = get_reranker().predict
    pairs, owner = [], []
    for i, p in enumerate(passages):
        for w in windows(p.text):
            pairs.append((question, w))
            owner.append(i)
    window_scores = [float(s) for s in score_pairs(pairs)]
    scores = [max(s for o, s in zip(owner, window_scores) if o == i) for i in range(len(passages))]
    if not all(math.isfinite(s) for s in window_scores):
        # Sorting by NaN silently keeps the input order and looks like a working rerank.
        log.warning("reranker returned non-finite scores; keeping retrieval order")
        return passages[:k]
    order = sorted(range(len(passages)), key=lambda i: -scores[i])  # stable on ties
    return [passages[i] for i in order[:k]]
