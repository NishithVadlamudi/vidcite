from api.answer import NOT_FOUND_MESSAGE, answer_question
from api.citations import resolve_citations
from api.prompts import NOT_FOUND, build_user_message
from retrieval.search import Passage

P = [
    Passage("vidA", "Raft talk", 3, 120.0, 165.5, "Raft elects a leader using randomized timeouts.", 0.82),
    Passage("vidA", "Raft talk", 4, 160.0, 204.0, "The leader replicates log entries to followers.", 0.74),
    Passage("vidB", "GFS talk", 9, 900.0, 941.0, "GFS uses a single master for metadata.", 0.61),
]


# --- citation mapping -------------------------------------------------------

def test_tags_map_to_stored_timestamps():
    text, cites, invalid = resolve_citations("Raft uses random timeouts [S1]. Logs flow from the leader [S2].", P)
    assert invalid == []
    assert [c["n"] for c in cites] == [1, 2]
    assert cites[0]["start_s"] == 120.0 and cites[0]["video_id"] == "vidA"
    assert cites[1]["start_s"] == 160.0


def test_citations_are_unique_in_order_of_first_use():
    _, cites, _ = resolve_citations("A [S3]. B [S1][S3]. C [S1].", P)
    assert [c["n"] for c in cites] == [3, 1]


def test_invalid_tags_are_removed_and_reported():
    text, cites, invalid = resolve_citations("Real claim [S1]. Made-up claim [S7].", P)
    assert invalid == [7]
    assert "[S7]" not in text and text.endswith("Made-up claim.")
    assert [c["n"] for c in cites] == [1]


# --- prompt -----------------------------------------------------------------

def test_prompt_numbers_passages_with_titles_and_times():
    msg = build_user_message("How does Raft pick a leader?", P)
    assert "[S1] Raft talk (2:00-2:45)" in msg
    assert "[S3] GFS talk (15:00-15:41)" in msg
    assert msg.rstrip().endswith("Question: How does Raft pick a leader?")


# --- full pipeline with fake database, embedder and LLM ----------------------

class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params):
        self.params = params

    def fetchall(self):
        return [(p.video_id, p.title, p.chunk_index, p.start_s, p.end_s, p.text, p.score) for p in self.rows]


class FakeConn:
    def __init__(self, rows):
        self.rows = rows

    def cursor(self):
        return FakeCursor(self.rows)


def fake_embed(q):
    return [0.0] * 384


def test_answer_with_citations():
    llm_calls = []

    def llm(msg):
        llm_calls.append(msg)
        return "Raft uses randomized election timeouts [S1]."

    r = answer_question(FakeConn(P), "How does Raft elect a leader?", complete=llm, embed=fake_embed)
    assert r["found"] is True and len(llm_calls) == 1
    assert r["citations"][0]["start_s"] == 120.0
    assert r["top_score"] == 0.82 and r["latency_ms"] >= 0


def test_weak_retrieval_skips_the_llm():
    weak = [Passage("vidB", "GFS talk", 1, 0.0, 40.0, "unrelated", 0.21)]

    def llm(msg):
        raise AssertionError("LLM should not be called")

    r = answer_question(FakeConn(weak), "What is the capital of France?", complete=llm, embed=fake_embed)
    assert r["found"] is False and r["answer"] == NOT_FOUND_MESSAGE and r["citations"] == []


def test_llm_says_not_found():
    r = answer_question(FakeConn(P), "Who founded Google?", complete=lambda m: NOT_FOUND, embed=fake_embed)
    assert r["found"] is False and r["answer"] == NOT_FOUND_MESSAGE


def test_answer_without_valid_citations_is_rejected():
    r = answer_question(FakeConn(P), "q?", complete=lambda m: "Confident but uncited claim [S9].", embed=fake_embed)
    assert r["found"] is False and r["citations"] == []


def test_empty_library():
    r = answer_question(FakeConn([]), "anything?", complete=lambda m: "x", embed=fake_embed)
    assert r["found"] is False and r["top_score"] is None
