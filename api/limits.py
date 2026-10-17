"""Usage limits for a public demo, so visitors can't exhaust the LLM quota.

- Per visitor: at most ASK_LIMIT_PER_HOUR questions per rolling hour (0 = no limit).
- Site-wide: at most LLM_DAILY_LIMIT questions per UTC day go to the LLM (0 = no limit);
  after that, /ask answers in search-only mode instead of failing.

In-memory, so limits reset when the server restarts; that's fine for one small
container. Both default to off for local use; the Dockerfile turns them on.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from datetime import datetime, timezone

ASK_LIMIT_PER_HOUR = int(os.environ.get("ASK_LIMIT_PER_HOUR", "0"))
LLM_DAILY_LIMIT = int(os.environ.get("LLM_DAILY_LIMIT", "0"))
WINDOW_S = 3600.0


class Limits:
    def __init__(self, per_visitor: int = ASK_LIMIT_PER_HOUR, llm_per_day: int = LLM_DAILY_LIMIT,
                 clock=time.time):
        self.per_visitor, self.llm_per_day, self.clock = per_visitor, llm_per_day, clock
        self._asks: dict[str, deque] = {}
        self._day, self._llm_used = None, 0
        self._lock = threading.Lock()  # sync endpoints run in a thread pool

    def visitor_wait_s(self, visitor: str) -> float:
        """0 if this visitor may ask now (and records the ask), else seconds until they can."""
        if not self.per_visitor:
            return 0.0
        now = self.clock()
        with self._lock:
            q = self._asks.setdefault(visitor, deque())
            while q and now - q[0] >= WINDOW_S:
                q.popleft()
            if len(q) >= self.per_visitor:
                return WINDOW_S - (now - q[0])
            q.append(now)
            if len(self._asks) > 10_000:  # forget idle visitors so memory stays bounded
                for k in [k for k, v in self._asks.items() if not v]:
                    del self._asks[k]
            return 0.0

    def take_llm_call(self) -> bool:
        """True if today's LLM budget allows another question (and counts it)."""
        if not self.llm_per_day:
            return True
        today = datetime.fromtimestamp(self.clock(), tz=timezone.utc).date()
        with self._lock:
            if today != self._day:
                self._day, self._llm_used = today, 0
            if self._llm_used >= self.llm_per_day:
                return False
            self._llm_used += 1
            return True


def visitor_id(headers, client_host: str | None) -> str:
    """Client address. Behind a proxy (e.g. Hugging Face Spaces) it's in X-Forwarded-For; the
    last entry is the one the proxy added, earlier ones can be set by the client. The daily LLM
    cap protects the quota even if this is spoofed."""
    forwarded = headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return client_host or "unknown"
