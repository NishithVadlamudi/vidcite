"""Parse YouTube captions (json3 format) into the same word list Whisper produces:
[{"text": "hello", "start": 0.0, "end": 0.4}, ...]

Two kinds of captions arrive in json3:
- Auto-generated: each event holds several segs, each a word with its own
  tOffsetMs, so we get real per-word timing.
- Manually uploaded: each event holds one seg with a whole line of text and
  no per-word timing, so we spread the line's duration across its words
  (accurate to a second or two, well inside our +/-15 s target).
"""
from __future__ import annotations

import re

NOISE = re.compile(r"^\[.*\]$")  # [Music], [Applause], [Laughter]


def _clean_tokens(text: str) -> list[str]:
    tokens = text.replace("\n", " ").split()
    return [t for t in tokens if not NOISE.match(t)]


def parse_json3(data: dict) -> list[dict]:
    raw: list[tuple[float, float, str]] = []  # (start_s, event_end_s, word)

    for event in data.get("events", []):
        segs = event.get("segs")
        if not segs or "tStartMs" not in event:
            continue
        ev_start = event["tStartMs"] / 1000
        ev_end = ev_start + event.get("dDurationMs", 0) / 1000

        has_word_timing = any("tOffsetMs" in s for s in segs)
        if has_word_timing:
            # Auto captions: one seg per word, offset relative to the event.
            for seg in segs:
                start = ev_start + seg.get("tOffsetMs", 0) / 1000
                for tok in _clean_tokens(seg.get("utf8", "")):
                    raw.append((start, ev_end, tok))
        else:
            # Manual captions: spread the line evenly across its words.
            tokens = _clean_tokens("".join(s.get("utf8", "") for s in segs))
            if not tokens:
                continue
            step = max(ev_end - ev_start, 0.01) / len(tokens)
            for i, tok in enumerate(tokens):
                raw.append((ev_start + i * step, ev_end, tok))

    raw.sort(key=lambda r: r[0])

    words: list[dict] = []
    for i, (start, ev_end, tok) in enumerate(raw):
        # A word ends when the next one starts, but never after its own event.
        nxt_start = raw[i + 1][0] if i + 1 < len(raw) else ev_end
        end = min(ev_end, nxt_start) if nxt_start > start else ev_end
        end = max(end, start + 0.05)
        words.append({"text": tok, "start": round(start, 2), "end": round(end, 2)})
    return words
