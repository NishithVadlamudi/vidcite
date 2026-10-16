"""The full question -> answer pipeline, independent of the web framework,
so the API, a CLI and the eval harness all call the same function."""
from __future__ import annotations

import logging
import time

from api.citations import resolve_citations
from api.prompts import NOT_FOUND, build_user_message
from retrieval.search import retrieve

log = logging.getLogger("answer")
NOT_FOUND_MESSAGE = "I couldn't find this in the video library."


def answer_question(conn, question: str, source: str = "youtube", video_id: str | None = None,
                    k: int = 5, complete=None, embed=None) -> dict:
    if complete is None:
        from api.llm import complete
    t0 = time.perf_counter()

    passages, found = retrieve(conn, question, k=k, source=source, video_id=video_id, embed=embed)
    result = {"answer": NOT_FOUND_MESSAGE, "citations": [], "found": False,
              "top_score": round(passages[0].score, 3) if passages else None}

    if found:
        raw = complete(build_user_message(question, passages))
        if raw.strip() != NOT_FOUND:
            text, citations, invalid = resolve_citations(raw, passages)
            if invalid:
                log.warning("dropped invalid citation tags %s for %r", invalid, question)
            if citations:  # an answer with no valid citation is treated as not found
                result.update(answer=text, citations=citations, found=True)

    result["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    return result
