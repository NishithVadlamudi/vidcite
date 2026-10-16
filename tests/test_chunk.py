from ingest.chunk import chunk_words, format_ts, is_punctuated, split_pseudo_sentences, split_sentences


def make_words(sentences, word_s=0.4, gap_s=0.1, pause_between=0.2):
    """Build fake Whisper words: each sentence is a list of words; the last
    word gets a period. Returns words with realistic timestamps."""
    words, t = [], 0.0
    for sent in sentences:
        for i, w in enumerate(sent):
            text = w + ("." if i == len(sent) - 1 else "")
            words.append({"text": text, "start": round(t, 2), "end": round(t + word_s, 2)})
            t += word_s + gap_s
        t += pause_between
    return words


def ten_word_sentences(n):
    return [[f"w{s}_{i}" for i in range(10)] for s in range(n)]  # ~5 s each


def test_empty_input():
    assert chunk_words([]) == []


def test_timestamps_come_from_words():
    words = make_words(ten_word_sentences(30))
    chunks = chunk_words(words)
    starts = {w["start"] for w in words}
    ends = {w["end"] for w in words}
    for c in chunks:
        assert c.start_s in starts and c.end_s in ends


def test_chunks_respect_max_and_reach_target():
    words = make_words(ten_word_sentences(60))  # ~5 minutes
    chunks = chunk_words(words, target_s=45, max_s=75)
    assert len(chunks) > 1
    for c in chunks:
        assert c.end_s - c.start_s <= 75
    for c in chunks[:-1]:
        assert c.end_s - c.start_s >= 45


def test_breaks_only_at_sentence_ends():
    words = make_words(ten_word_sentences(30))
    for c in chunk_words(words):
        assert c.text.endswith(".")


def test_consecutive_chunks_overlap_by_one_sentence():
    words = make_words(ten_word_sentences(30))
    chunks = chunk_words(words)
    for a, b in zip(chunks, chunks[1:]):
        last_sentence_of_a = a.text.split(". ")[-1]
        assert b.text.startswith(last_sentence_of_a.rstrip("."))
        assert b.start_s < a.end_s


def test_no_overlap_option():
    words = make_words(ten_word_sentences(30))
    chunks = chunk_words(words, overlap=False)
    for a, b in zip(chunks, chunks[1:]):
        assert b.start_s > a.end_s


def test_every_word_is_covered():
    words = make_words(ten_word_sentences(40))
    text = " ".join(c.text for c in chunk_words(words))
    for w in words:
        assert w["text"].rstrip(".") in text


def test_long_pause_is_a_boundary_without_punctuation():
    words = [
        {"text": "so", "start": 0.0, "end": 0.3},
        {"text": "anyway", "start": 0.4, "end": 0.8},
        {"text": "next", "start": 2.0, "end": 2.3},  # 1.2 s pause before
    ]
    sents = split_sentences(words, pause_s=0.6)
    assert [s.text for s in sents] == ["so anyway", "next"]


def test_runaway_sentence_is_capped():
    words = [{"text": f"w{i}", "start": i * 0.5, "end": i * 0.5 + 0.4} for i in range(400)]  # 200 s, no punctuation
    for c in chunk_words(words, target_s=45, max_s=75):
        assert c.end_s - c.start_s <= 75


def test_short_transcript_is_one_chunk():
    words = make_words(ten_word_sentences(2))
    chunks = chunk_words(words)
    assert len(chunks) == 1 and chunks[0].start_s == 0.0


def test_format_ts():
    assert format_ts(5) == "0:05"
    assert format_ts(1421.3) == "23:41"
    assert format_ts(3725) == "1:02:05"


# --- unpunctuated transcripts (some YouTube auto-captions) ---------------------

def unpunctuated(n, word_s=0.3):
    """Caption-style words: no punctuation, each word ends where the next starts."""
    return [{"text": f"w{i}", "start": round(i * word_s, 2), "end": round((i + 1) * word_s, 2)} for i in range(n)]


def test_punctuation_detection():
    assert is_punctuated(make_words(ten_word_sentences(5)))
    assert not is_punctuated(unpunctuated(500))


def test_pseudo_sentences_stay_between_5_and_15_seconds():
    sents = split_pseudo_sentences(unpunctuated(1000))  # 300 s
    for s in sents[:-1]:
        assert 5.0 <= s.end - s.start <= 15.0


def test_pseudo_sentences_prefer_the_longest_pause():
    words = unpunctuated(60)  # 18 s
    for w in words[30:]:  # a 1 s pause after word 29 (9 s in)
        w["start"] += 1.0
        w["end"] += 1.0
    assert split_pseudo_sentences(words)[0].words[-1]["text"] == "w29"


def test_unpunctuated_transcript_gets_normal_chunks_with_overlap():
    chunks = chunk_words(unpunctuated(2000))  # 600 s, like the Figma captions
    for c in chunks[:-1]:
        assert 45 <= c.end_s - c.start_s <= 75
    for a, b in zip(chunks, chunks[1:]):
        assert b.start_s < a.end_s  # one pseudo-sentence of overlap
