from api.answer import answer_question
from api.citations import claims_for
from retrieval.moment import LINK_LEAD_S, best_moment, sentence_times
from retrieval.search import Passage, rrf, top_score


def p(idx, score=0.5, video="vidA", start=0.0, end=10.0, text="x."):
    return Passage(video, "talk", idx, start, end, text, score)


# --- rank fusion ---------------------------------------------------------------

def test_rrf_rewards_passages_found_by_both_methods():
    vector = [p(1), p(2), p(3)]
    keyword = [p(3), p(4)]
    fused = rrf([vector, keyword], k=4)
    # 3 is rank 3 + rank 1, which beats 1 (rank 1 in one list only)
    assert [x.chunk_index for x in fused] == [3, 1, 2, 4]


def test_rrf_dedupes_and_truncates():
    fused = rrf([[p(1), p(2)], [p(1), p(2)]], k=1)
    assert [x.chunk_index for x in fused] == [1]


def test_rrf_keeps_chunks_from_different_videos_apart():
    fused = rrf([[p(1, video="A")], [p(1, video="B")]], k=5)
    assert len(fused) == 2


def test_top_score_is_best_cosine_not_first_passage():
    assert top_score([p(1, score=0.6), p(2, score=0.8)]) == 0.8


# --- sentence-level moments ----------------------------------------------------

CHUNK = Passage("vidA", "talk", 0, 100.0, 120.0,
                "Intro words here now. HNSW is faster on query. It uses more memory too.", 0.9)


def test_sentence_times_spread_by_word_count():
    times = sentence_times(CHUNK)  # 4 + 5 + 5 = 14 words over 20 s
    assert [round(t, 2) for t, _ in times] == [100.0, round(100 + 20 * 4 / 14, 2), round(100 + 20 * 9 / 14, 2)]
    assert times[1][1] == "HNSW is faster on query."


def test_best_moment_picks_most_similar_sentence():
    def embed_texts(texts):
        return [[1.0, 0.0] if "HNSW" in t else [0.0, 1.0] for t in texts]

    assert best_moment(CHUNK, [1.0, 0.0], embed_texts, lead_s=0) == round(100 + 20 * 4 / 14, 2)
    assert best_moment(CHUNK, [1.0, 0.0], embed_texts, lead_s=3) == round(100 + 20 * 4 / 14 - 3, 2)


def test_lead_in_never_goes_before_the_video_start():
    early = Passage("vidA", "talk", 0, 0.0, 10.0, "One two. Three four.", 0.9)
    assert best_moment(early, [1.0], lambda texts: [[0.0], [1.0]], lead_s=8) == 0.0


def test_best_moment_single_sentence_falls_back_to_chunk_start():
    one = Passage("vidA", "talk", 0, 50.0, 60.0, "Just one sentence", 0.9)
    assert best_moment(one, [1.0], lambda texts: [[1.0]]) == 50.0


def test_claims_for_collects_sentences_citing_a_passage():
    answer = "Use HNSW [S1]. It is faster [S2]. Match the operator class [S1][S2]."
    assert claims_for(answer, 1) == "Use HNSW. Match the operator class."
    assert claims_for(answer, 3) == ""


# --- answers carry a link_s ----------------------------------------------------

class FakeCursor:
    def __init__(self, rows): self.rows = rows
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, sql, params): pass
    def fetchall(self):
        return [(x.video_id, x.title, x.chunk_index, x.start_s, x.end_s, x.text, x.score) for x in self.rows]


class FakeConn:
    def __init__(self, rows): self.rows = rows
    def cursor(self): return FakeCursor(self.rows)


def test_answer_citations_link_to_the_matching_sentence():
    def embed(text):
        return [1.0, 0.0] if "HNSW" in text or "faster" in text else [0.0, 1.0]

    r = answer_question(FakeConn([CHUNK]), "Which index?", complete=lambda m: "HNSW is faster [S1].",
                        embed=embed, hybrid=True)
    c = r["citations"][0]
    assert c["start_s"] == 100.0  # chunk span unchanged
    assert c["link_s"] == round(100 + 20 * 4 / 14 - LINK_LEAD_S, 2)  # HNSW sentence, minus lead-in
