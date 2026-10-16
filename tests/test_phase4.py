from api.answer import answer_question
from api.citations import claims_for
from retrieval.moment import LINK_LEAD_S, best_moment, sentence_times
from retrieval.search import Passage, merge_neighbours, retrieve, rrf, top_score


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


# --- neighbour merging ---------------------------------------------------------

def test_merge_joins_adjacent_chunks_and_drops_the_shared_sentence():
    a = Passage("vidA", "talk", 4, 100.0, 145.0, "One. Two. Shared.", 0.6)
    b = Passage("vidA", "talk", 5, 141.0, 190.0, "Shared. Three.", 0.8)
    far = Passage("vidA", "talk", 9, 400.0, 440.0, "Far away.", 0.7)
    merged = merge_neighbours([b, far, a])  # rank order: b first
    assert len(merged) == 2
    m = merged[0]
    assert (m.chunk_index, m.start_s, m.end_s) == (4, 100.0, 190.0)
    assert m.text == "One. Two. Shared. Three."
    assert m.score == 0.8
    # "Three." is timed inside chunk 5 (141-190 s), not spread over the merged 100-190 s
    assert sentence_times(m)[-1] == (141.0 + 49.0 * 1 / 2, "Three.")
    assert merged[1] is far


def test_merge_never_joins_different_videos():
    a = Passage("vidA", "talk", 4, 0.0, 10.0, "A.", 0.6)
    b = Passage("vidB", "talk", 5, 0.0, 10.0, "B.", 0.6)
    assert merge_neighbours([a, b]) == [a, b]


def test_rrf_keeps_the_all_terms_flag_from_keyword_results():
    vec = [p(1)]
    kw = [Passage("vidA", "talk", 1, 0.0, 10.0, "x.", 0.5, all_terms=True)]
    assert rrf([vec, kw], k=1)[0].all_terms is True


# --- keyword exception to the confidence gate ----------------------------------

class KeywordCursor(FakeCursor):
    def fetchall(self):
        return [(x.video_id, x.title, x.chunk_index, x.start_s, x.end_s, x.text, x.score, x.all_terms)
                for x in self.rows] if self.sql_has_keyword else super().fetchall()

    def execute(self, sql, params):
        self.sql_has_keyword = "plainto_tsquery" in sql


class KeywordConn(FakeConn):
    def cursor(self): return KeywordCursor(self.rows)


def test_weak_cosine_passes_gate_when_every_word_matches():
    lsn = Passage("vidA", "talk", 16, 700.0, 750.0, "The LSN marks a point.", 0.51, all_terms=True)
    _, found = retrieve(KeywordConn([lsn]), "What is an LSN?", embed=lambda q: [0.0], hybrid=True)
    assert found is True
    _, found = retrieve(KeywordConn([lsn]), "What is an LSN?", embed=lambda q: [0.0], hybrid=False)
    assert found is False  # vector-only: no keyword evidence, 0.51 < 0.55


def test_unpunctuated_chunk_is_split_into_word_blocks():
    words = " ".join(f"w{i}" for i in range(45))  # no punctuation, 45 words over 45 s
    chunk = Passage("vidA", "talk", 0, 0.0, 45.0, words, 0.9)
    times = sentence_times(chunk)
    assert [t for t, _ in times] == [0.0, 15.0, 30.0]
    assert times[1][1].startswith("w15 ")


# --- reranking -----------------------------------------------------------------

def test_rerank_reorders_by_cross_encoder_score_and_keeps_cosine():
    from retrieval.rerank import rerank
    passages = [p(1, score=0.9, text="summary of everything."), p(2, score=0.6, text="the exact answer.")]
    scores = lambda pairs: [5.0 if "exact" in text else 1.0 for _, text in pairs]
    out = rerank("q?", passages, k=1, score_pairs=scores)
    assert [x.chunk_index for x in out] == [2]
    assert out[0].score == 0.6  # cosine kept for the confidence gate


def test_retrieve_reranks_a_wider_pool_down_to_k():
    rows = [p(i, score=0.7, text=f"chunk {i}.") for i in range(20)]
    seen = []

    def scores(pairs):
        seen.append(len(pairs))
        return [float(text == "chunk 17.") for _, text in pairs]

    out, found = retrieve(FakeConn(rows), "q?", k=5, embed=lambda q: [0.0], hybrid=False,
                          rerank=True, score_pairs=scores)
    assert seen == [20]  # reranker saw the 20-candidate pool, not just the top 5
    assert out[0].chunk_index == 17 and len(out) == 5 and found


def test_rerank_with_nan_scores_keeps_retrieval_order(caplog):
    from retrieval.rerank import rerank
    passages = [p(i) for i in range(4)]
    out = rerank("q?", passages, k=2, score_pairs=lambda pairs: [float("nan")] * len(pairs))
    assert [x.chunk_index for x in out] == [0, 1]
    assert "non-finite" in caplog.text
