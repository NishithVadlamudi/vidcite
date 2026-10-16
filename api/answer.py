"""The full question -> answer pipeline, independent of the web framework,
so the API, a CLI and the eval harness all call the same function."""
from __future__ import annotations

import logging
import time

from api.citations import claims_for, resolve_citations
from api.prompts import NOT_FOUND, build_user_message
from retrieval.moment import best_moment
from retrieval.search import retrieve, top_score

log = logging.getLogger("answer")
NOT_FOUND_MESSAGE = "I couldn't find this in the video library."


def answer_question(conn, question: str, source: str = "youtube", video_id: str | None = None,
                    k: int = 5, complete=None, embed=None, embed_texts=None,
                    hybrid: bool | None = None, merge: bool | None = None) -> dict:
    if complete is None:
        from api.llm import complete
    t0 = time.perf_counter()

    passages, found = retrieve(conn, question, k=k, source=source, video_id=video_id, embed=embed,
                               hybrid=hybrid, merge=merge)
    result = {"answer": NOT_FOUND_MESSAGE, "citations": [], "found": False,
              "top_score": round(top_score(passages), 3) if passages else None}

    if found:
        raw = complete(build_user_message(question, passages))
        if raw.strip() != NOT_FOUND:
            text, citations, invalid = resolve_citations(raw, passages)
            if invalid:
                log.warning("dropped invalid citation tags %s for %r", invalid, question)
            if citations:  # an answer with no valid citation is treated as not found
                add_links(citations, passages, text, question, embed, embed_texts)
                result.update(answer=text, citations=citations, found=True)

    result["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    return result


def add_links(citations, passages, answer: str, question: str, embed=None, embed_texts=None) -> None:
    """Set each citation's link_s to the sentence in its chunk that best matches the
    claim it supports. start_s/end_s stay the chunk span."""
    if embed_texts is None:
        if embed is not None:  # injected (tests): reuse it for sentences too
            embed_texts = lambda texts: [embed(t) for t in texts]
        else:
            from ingest.embed import QUERY_DEVICE, embed_passages, get_model

            embed_texts = lambda texts: embed_passages(texts, model=get_model(QUERY_DEVICE))
    if embed is None:
        from ingest.embed import embed_query as embed
    for c in citations:
        claim = claims_for(answer, c["n"]) or question
        c["link_s"] = best_moment(passages[c["n"] - 1], embed(claim), embed_texts)
