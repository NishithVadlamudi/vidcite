"""Turn word-level transcripts into time-stamped chunks.

Input: words like {"text": "Hello", "start": 0.0, "end": 0.4} (Whisper output).
Output: chunks of roughly `target_s` seconds that only break at sentence ends
or long pauses, never longer than `max_s`, each overlapping the previous chunk
by one sentence so answers that span a boundary still appear whole.

Some YouTube auto-captions have no punctuation at all, and caption word timings
carry no pauses (each word ends where the next starts). Splitting those at "sentence
ends or long pauses" only produces forced cuts at max_s, so unpunctuated transcripts
use pseudo-sentences instead: cut at the longest pause 5-15 s in, which with no pause
information means roughly every 5 s.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

SENTENCE_END = (".", "?", "!")
# Sentence-end marks per word below which a transcript counts as unpunctuated.
# Punctuated captions and Whisper run ~0.06-0.09; the unpunctuated Figma captions were 0.000.
MIN_PUNCTUATION_RATE = 0.02
PSEUDO_SENTENCE_S = (5.0, 15.0)


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


def is_punctuated(words: list[dict]) -> bool:
    ends = sum(w["text"].strip().endswith(SENTENCE_END) for w in words)
    return ends / len(words) >= MIN_PUNCTUATION_RATE


def split_pseudo_sentences(words: list[dict], min_s: float = PSEUDO_SENTENCE_S[0],
                           max_s: float = PSEUDO_SENTENCE_S[1]) -> list[Sentence]:
    """For unpunctuated transcripts: cut each pseudo-sentence at the longest pause
    between min_s and max_s after it starts (earliest on ties)."""
    sentences: list[Sentence] = []
    i = 0
    while i < len(words):
        j = i  # last word that still fits within max_s
        while j + 1 < len(words) and words[j + 1]["end"] - words[i]["start"] <= max_s:
            j += 1
        if j + 1 >= len(words):
            sentences.append(Sentence(words[i:]))
            break
        candidates = [k for k in range(i, j) if words[k]["end"] - words[i]["start"] >= min_s] or [j]
        cut = max(candidates, key=lambda k: words[k + 1]["start"] - words[k]["end"])
        sentences.append(Sentence(words[i:cut + 1]))
        i = cut + 1
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

    if is_punctuated(words):
        sentences = split_sentences(words, pause_s=pause_s, max_s=max_s)
    else:
        sentences = split_pseudo_sentences(words)
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
