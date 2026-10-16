"""The one place that talks to the LLM, so swapping providers is a one-file change."""
from __future__ import annotations

import os

from api.prompts import SYSTEM

MODEL = os.environ.get("LLM_MODEL", "gemini-3.5-flash-lite")

_client = None


class LLMUnavailable(RuntimeError):
    """The provider refused or failed the request (overloaded, bad key, unknown model...).
    Callers catch this instead of provider-specific errors."""


def complete(user_message: str) -> str:
    global _client
    if _client is None:
        from dotenv import load_dotenv
        from google import genai

        load_dotenv()
        _client = genai.Client()  # reads GEMINI_API_KEY
    from google.genai import errors, types

    try:
        response = _client.models.generate_content(
            model=MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,
                max_output_tokens=400,
                temperature=0,
                # Thinking tokens count against max_output_tokens; keep them minimal so they can't
                # crowd out the answer. Gemini 3.x uses thinking_level (thinking_budget is 2.5-only).
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            ),
        )
    except errors.APIError as e:
        raise LLMUnavailable(f"{MODEL} returned {e.code} {e.status}: {e.message}") from e
    return (response.text or "").strip()  # text is None if the response was blocked
