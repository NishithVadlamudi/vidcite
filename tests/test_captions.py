from ingest.captions import parse_json3
from ingest.chunk import chunk_words

AUTO = {  # shape of YouTube auto-generated json3: per-word tOffsetMs
    "events": [
        {"tStartMs": 0, "dDurationMs": 200000},  # window definition, no segs
        {"tStartMs": 1000, "dDurationMs": 3000, "segs": [
            {"utf8": "so"}, {"utf8": " today", "tOffsetMs": 400}, {"utf8": " we", "tOffsetMs": 900},
            {"utf8": " talk", "tOffsetMs": 1300}, {"utf8": " about", "tOffsetMs": 1700}, {"utf8": " raft", "tOffsetMs": 2100}]},
        {"tStartMs": 3900, "aAppend": 1, "dDurationMs": 100, "segs": [{"utf8": "\n"}]},
        {"tStartMs": 4000, "dDurationMs": 2500, "segs": [
            {"utf8": "[Music]"}]},
        {"tStartMs": 7000, "dDurationMs": 2000, "segs": [
            {"utf8": "leader"}, {"utf8": " election", "tOffsetMs": 500}]},
    ]
}

MANUAL = {  # shape of manually uploaded json3: one line per event, no offsets
    "events": [
        {"tStartMs": 0, "dDurationMs": 4000, "segs": [{"utf8": "Today we talk about Raft."}]},
        {"tStartMs": 4000, "dDurationMs": 2000, "segs": [{"utf8": "It elects a leader."}]},
    ]
}


def test_auto_captions_keep_per_word_timing():
    words = parse_json3(AUTO)
    assert [w["text"] for w in words] == ["so", "today", "we", "talk", "about", "raft", "leader", "election"]
    assert words[1]["start"] == 1.4
    assert words[6]["start"] == 7.0 and words[7]["start"] == 7.5


def test_noise_and_newlines_are_dropped():
    texts = [w["text"] for w in parse_json3(AUTO)]
    assert "[Music]" not in texts and "" not in texts


def test_word_ends_at_next_start_but_not_after_its_event():
    words = parse_json3(AUTO)
    assert words[0]["end"] == words[1]["start"]
    assert words[5]["end"] == 4.0  # "raft" event ends at 4.0, next word is at 7.0


def test_manual_captions_spread_across_line():
    words = parse_json3(MANUAL)
    assert [w["text"] for w in words][:5] == ["Today", "we", "talk", "about", "Raft."]
    assert words[0]["start"] == 0.0 and words[4]["start"] == 3.2
    assert words[5]["start"] == 4.0


def test_times_are_monotonic_and_valid():
    for data in (AUTO, MANUAL):
        words = parse_json3(data)
        for a, b in zip(words, words[1:]):
            assert a["start"] <= b["start"]
        for w in words:
            assert w["end"] > w["start"]


def test_empty_input():
    assert parse_json3({}) == [] and parse_json3({"events": []}) == []


def test_captions_flow_into_chunker():
    chunks = chunk_words(parse_json3(MANUAL))
    assert len(chunks) == 1
    assert chunks[0].start_s == 0.0 and chunks[0].text.endswith("leader.")
