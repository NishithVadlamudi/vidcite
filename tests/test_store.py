from contextlib import contextmanager

import pytest

from ingest.store import DELETE_CHUNKS, INSERT_CHUNK, UPSERT_VIDEO, store_video


class FakeCursor:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params):
        self.log.append(("execute", sql, params))

    def executemany(self, sql, rows):
        self.log.append(("executemany", sql, list(rows)))


class FakeConn:
    def __init__(self):
        self.log = []

    @contextmanager
    def transaction(self):
        self.log.append("begin")
        yield
        self.log.append("commit")

    def cursor(self):
        return FakeCursor(self.log)


META = {"id": "vid1", "title": "Raft talk", "channel": "c", "duration_s": 3600, "license": "Creative Commons"}
CHUNKS = [
    {"index": 0, "start_s": 0.0, "end_s": 45.2, "text": "first"},
    {"index": 1, "start_s": 40.1, "end_s": 88.0, "text": "second"},
]


def test_replaces_chunks_for_that_source_in_one_transaction():
    conn = FakeConn()
    n = store_video(conn, META, "youtube", CHUNKS, [[0.1] * 384, [0.2] * 384])
    assert n == 2
    assert conn.log[0] == "begin" and conn.log[-1] == "commit"
    ops = [entry[1] for entry in conn.log[1:-1]]
    assert ops == [UPSERT_VIDEO, DELETE_CHUNKS, INSERT_CHUNK]  # delete before insert
    assert conn.log[2][2] == ("vid1", "youtube")


def test_rows_carry_source_and_exact_timestamps():
    conn = FakeConn()
    store_video(conn, META, "whisper", CHUNKS, [[0.1] * 384, [0.2] * 384])
    rows = conn.log[3][2]
    assert rows[0][:6] == ("vid1", "whisper", 0, 0.0, 45.2, "first")
    assert rows[1][:6] == ("vid1", "whisper", 1, 40.1, 88.0, "second")


def test_mismatched_vectors_rejected_before_writing():
    conn = FakeConn()
    with pytest.raises(ValueError):
        store_video(conn, META, "youtube", CHUNKS, [[0.1] * 384])
    assert conn.log == []
