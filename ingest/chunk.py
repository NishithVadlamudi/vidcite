"""Turn word-level transcripts into time-stamped chunks.

Input: words like {"text": "Hello", "start": 0.0, "end": 0.4} (Whisper output).
Output: chunks of roughly `target_s` seconds that only break at sentence ends
or long pauses, never longer than `max_s`, each overlapping the previous chunk
by one sentence so answers that span a boundary still appear whole.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

SENTENCE_END = (".", "?", "!")


@dataclass
class Sentence:
    words: list[dict]

    @property
    def start(self) -> float:
        return self.words[0]["start"]

    @property
    def end(self) -> float:
        return self.words[-1]["end"]

    @property
    def text(self) -> str:
        return " ".join(w["text"].strip() for w in self.words).strip()


@dataclass
class Chunk:
    index: int
    start_s: float
    end_s: float
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def split_sentences(words: list[dict], pause_s: float = 0.6, max_s: float = 75.0) -> list[Sentence]:
    """Group words into sentences. A long pause also counts as a boundary,
    because Whisper often drops punctuation in rambling speech."""
    sentences: list[Sentence] = []
    current: list[dict] = []
    for i, word in enumerate(words):
        current.append(word)
        nxt = words[i + 1] if i + 1 < len(words) else None
        ends_sentence = word["text"].strip().endswith(SENTENCE_END)
        long_pause = nxt is not None and nxt["start"] - word["end"] >= pause_s
        too_long = nxt is not None and nxt["end"] - current[0]["start"] > max_s
        if ends_sentence or long_pause or too_long or nxt is None:
            sentences.append(Sentence(current))
            current = []
    return sentences


def chunk_words(
    words: list[dict],
    target_s: float = 45.0,
    max_s: float = 75.0,
    pause_s: float = 0.6,
    overlap: bool = True,
) -> list[Chunk]:
    """Build chunks from word-level timestamps. Timestamps are copied from the
    words themselves, so they are exact, never estimated."""
    if not words:
        return []

    sentences = split_sentences(words, pause_s=pause_s, max_s=max_s)
    chunks: list[Chunk] = []
    current: list[Sentence] = []
    new_in_current = False  # does `current` hold anything beyond the overlap sentence?

    def close() -> None:
        nonlocal current, new_in_current
        chunks.append(
            Chunk(
                index=len(chunks),
                start_s=round(current[0].start, 2),
                end_s=round(current[-1].end, 2),
                text=" ".join(s.text for s in current),
            )
        )
        last = current[-1]
        # Carry the last sentence forward as overlap, unless it is itself huge.
        current = [last] if overlap and (last.end - last.start) < target_s else []
        new_in_current = False

    for sentence in sentences:
        if current and sentence.end - current[0].start > max_s:
            if new_in_current:
                close()
            # Overlap plus this sentence would still be too long: drop the overlap.
            if current and sentence.end - current[0].start > max_s:
                current = []
        current.append(sentence)
        new_in_current = True
        if current[-1].end - current[0].start >= target_s:
            close()

    if current and new_in_current:
        close()
    return chunks


def format_ts(seconds: float) -> str:
    """1421.3 -> '23:41' (or '1:02:05' for long videos)."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"
