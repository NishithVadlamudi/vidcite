"""Ingest one or more videos:

    python -m ingest.cli VIDEO_ID [VIDEO_ID ...] --source youtube|whisper [--own]

Writes data/chunks/<video_id>.<source>.json, then embeds the chunks and stores
them in Supabase. Use --no-store to stop after chunking (no database needed).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ingest.captions import parse_json3
from ingest.chunk import chunk_words, format_ts
from ingest.download import download_audio, download_captions, fetch_metadata
from yt_dlp.utils import DownloadError

CHUNK_DIR = Path("data/chunks")


def get_words(video_id: str, meta: dict, source: str) -> list[dict]:
    if source == "youtube":
        captions = download_captions(video_id, meta)
        meta["transcript_source"] = f"youtube-{captions['kind']}"
        return parse_json3(captions["data"])
    from ingest.transcribe import transcribe  # imported lazily: heavy

    meta["transcript_source"] = "whisper"
    return transcribe(video_id, download_audio(video_id))


def ingest(video_id: str, source: str, own: bool = False, conn=None) -> dict:
    meta = fetch_metadata(video_id, own=own)
    words = get_words(video_id, meta, source)
    chunks = chunk_words(words)

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    out = CHUNK_DIR / f"{video_id}.{source}.json"
    out.write_text(json.dumps({"video": meta, "chunks": [c.to_dict() for c in chunks]}, indent=1))

    print(f"{meta['title'][:60]}")
    print(f"  source={meta['transcript_source']}  words={len(words)}  chunks={len(chunks)}  -> {out}")
    if chunks:
        c = chunks[0]
        print(f"  first chunk [{format_ts(c.start_s)}-{format_ts(c.end_s)}]: {c.text[:90]}...")

    if conn is not None and chunks:
        from ingest.embed import embed_passages
        from ingest.store import store_video

        vectors = embed_passages([c.text for c in chunks])
        db_meta = {k: meta[k] for k in ("id", "title", "channel", "duration_s", "license")}
        n = store_video(conn, db_meta, source, [c.to_dict() for c in chunks], vectors)
        print(f"  stored {n} chunks in Supabase")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Creative Commons YouTube videos")
    parser.add_argument("video_ids", nargs="+")
    parser.add_argument("--source", choices=["youtube", "whisper"], default="youtube")
    parser.add_argument("--own", action="store_true", help="videos you uploaded yourself")
    parser.add_argument("--no-store", action="store_true", help="chunk only; skip embedding and database")
    args = parser.parse_args()

    conn = None
    if not args.no_store:
        from ingest.store import connect

        conn = connect()

    try:
        for vid in args.video_ids:
            try:
                ingest(vid, args.source, own=args.own, conn=conn)
            except (PermissionError, FileNotFoundError, DownloadError) as e:
                # One bad video (not CC, no captions, YouTube 429) shouldn't abort the batch.
                print(f"SKIPPED {vid}: {e}")
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()
