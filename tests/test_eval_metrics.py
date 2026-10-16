from evals.metrics import (best_threshold, contains, first_hit_rank, gate_scores,
                           link_error, on_time, summarize_retrieval)
from retrieval.search import Passage

P = [
    Passage("vidA", "Raft talk", 3, 120.0, 165.5, "Raft elects a leader.", 0.82),
    Passage("vidA", "Raft talk", 4, 160.0, 204.0, "The leader replicates logs.", 0.74),
    Passage("vidB", "GFS talk", 9, 900.0, 941.0, "GFS uses a single master.", 0.61),
]


# --- retrieval ---------------------------------------------------------------

def test_contains_needs_same_video_and_time_inside_span():
    assert contains(P[0], "vidA", 120.0) and contains(P[0], "vidA", 165.5)
    assert not contains(P[0], "vidA", 166.0)
    assert not contains(P[0], "vidB", 130.0)


def test_first_hit_rank_is_one_based_and_uses_first_match():
    assert first_hit_rank(P, "vidA", 162.0) == 1  # in both overlapping chunks
    assert first_hit_rank(P, "vidA", 190.0) == 2
    assert first_hit_rank(P, "vidB", 10.0) is None


def test_link_error_is_how_early_the_link_jumps():
    assert link_error(160.0, 190.0) == 30.0


def test_on_time_window_is_lopsided():
    assert on_time(0.0) and on_time(15.0) and on_time(-3.0)
    assert not on_time(15.1)   # too early: viewer waits through unrelated speech
    assert not on_time(-3.1)   # too late: viewer misses the start of the answer


def test_summarize_retrieval():
    rows = [{"rank": 1, "link_err": 5.0, "moment_err": -1.0},
            {"rank": 2, "link_err": 30.0, "moment_err": -8.0},
            {"rank": None, "link_err": None}]
    s = summarize_retrieval(rows, k=5)
    assert s["n"] == 3
    assert s["hit@1"] == 1 / 3 and s["hit@5"] == 2 / 3
    assert s["mrr"] == (1 + 0.5) / 3
    assert s["link_on_time"] == 0.5  # 5 s is on time, 30 s is too early
    assert s["link_abs_median_s"] == 17.5 and s["link_abs_max_s"] == 30.0
    assert s["moment_on_time"] == 0.5  # 1 s late is fine, 8 s late is not
    assert s["moment_abs_max_s"] == 8.0


# --- confidence gate ---------------------------------------------------------

GATE = [
    {"answerable": True, "top_score": 0.80},
    {"answerable": True, "top_score": 0.62},
    {"answerable": False, "top_score": 0.58},
    {"answerable": False, "top_score": 0.40},
]


def test_gate_scores():
    g = gate_scores(GATE, 0.60)
    assert g["pass_rate"] == 1.0 and g["reject_rate"] == 1.0 and g["balanced"] == 1.0
    g = gate_scores(GATE, 0.55)
    assert g["pass_rate"] == 1.0 and g["reject_rate"] == 0.5 and g["balanced"] == 0.75


def test_best_threshold_picks_middle_of_tied_range():
    # 0.59, 0.60, 0.61, 0.62 all separate perfectly; the middle one is chosen.
    best = best_threshold(GATE, [0.55, 0.59, 0.60, 0.61, 0.62, 0.65])
    assert best["balanced"] == 1.0 and best["threshold"] == 0.61
