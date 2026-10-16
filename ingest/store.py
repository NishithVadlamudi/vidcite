"""Write a video and its embedded chunks to Postgres (Supabase).

Re-ingesting the same video + source replaces its chunks atomically, so the
database never holds a half-written video.
"""
from __future__ import annotations

import os

UPSERT_VIDEO = """
insert into videos (id, title, channel, duration_s, license)
values (%(id)s, %(title)s, %(channel)s, %(duration_s)s, %(license)s)
on conflict (id) do update set
  title = excluded.title, channel = excluded.channel,
  duration_s = excluded.duration_s, license = excluded.license,
  ingested_at = now()
"""

DELETE_CHUNKS = "delete from chunks where video_id = %s and source = %s"

INSERT_CHUNK = """
insert into chunks (video_id, source, chunk_index, start_s, end_s, text, embedding)
values (%s, %s, %s, %s, %s, %s, %s)
"""


def connect():
    import psycopg
    from dotenv import load_dotenv
    from pgvector.psycopg import register_vector

    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("Set DATABASE_URL in .env (Supabase > Project Settings > Database > Connection string)")
    conn = psycopg.connect(url)
    register_vector(conn)
    return conn


def store_video(conn, meta: dict, source: str, chunks: list[dict], vectors: list[list[float]]) -> int:
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    rows = [
        (meta["id"], source, c["index"], c["start_s"], c["end_s"], c["text"], _as_vector(v))
        for c, v in zip(chunks, vectors)
    ]
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(UPSERT_VIDEO, meta)
            cur.execute(DELETE_CHUNKS, (meta["id"], source))
            cur.executemany(INSERT_CHUNK, rows)
    return len(rows)


def _as_vector(v):
    try:
        import numpy as np

        return np.asarray(v, dtype=np.float32)
    except ImportError:  # tests without numpy
        return v
