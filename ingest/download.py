"""Fetch metadata, captions and audio for a Creative Commons YouTube video.

Every function refuses videos that aren't Creative Commons unless own=True
(for videos you uploaded yourself). The license is stored with the video.
"""
from __future__ import annotations

import json
from pathlib import Path

import yt_dlp

AUDIO_DIR = Path("data/audio")
CAPTION_DIR = Path("data/captions")


def _url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def fetch_metadata(video_id: str, own: bool = False) -> dict:
    with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(_url(video_id), download=False)
    meta = {
        "id": video_id,
        "title": info.get("title", video_id),
        "channel": info.get("channel"),
        "duration_s": info.get("duration"),
        "license": info.get("license") or "",
        "has_manual_en": "en" in (info.get("subtitles") or {}),
        "has_auto_en": "en" in (info.get("automatic_captions") or {}),
    }
    if not own and "creative commons" not in meta["license"].lower():
        raise PermissionError(
            f"{video_id} is not Creative Commons (license: {meta['license'] or 'standard'}). "
            "Use YouTube's Creative Commons search filter, or pass own=True for your own videos."
        )
    if own and not meta["license"]:
        meta["license"] = "owned by uploader"
    return meta


def download_captions(video_id: str, meta: dict) -> dict:
    """Download English captions in json3. Prefers auto-generated captions,
    because they carry per-word timing; falls back to manual ones."""
    CAPTION_DIR.mkdir(parents=True, exist_ok=True)
    if meta["has_auto_en"]:
        kind, opts = "auto", {"writeautomaticsub": True}
    elif meta["has_manual_en"]:
        kind, opts = "manual", {"writesubtitles": True}
    else:
        raise FileNotFoundError(f"{video_id} has no English captions; use --source whisper")

    out = CAPTION_DIR / f"{video_id}.{kind}.json"
    if not out.exists():
        opts.update({
            "quiet": True,
            "skip_download": True,
            "subtitleslangs": ["en"],
            "subtitlesformat": "json3",
            "outtmpl": str(CAPTION_DIR / f"{video_id}.{kind}"),
        })
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([_url(video_id)])
        written = CAPTION_DIR / f"{video_id}.{kind}.en.json3"
        written.rename(out)
    return {"kind": kind, "data": json.loads(out.read_text())}


def download_audio(video_id: str) -> str:
    """16 kHz mono WAV for Whisper. Call fetch_metadata first (license check)."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    out = AUDIO_DIR / f"{video_id}.wav"
    if not out.exists():
        opts = {
            "quiet": True,
            "format": "bestaudio/best",
            "outtmpl": str(AUDIO_DIR / f"{video_id}.%(ext)s"),
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
            "postprocessor_args": ["-ar", "16000", "-ac", "1"],
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([_url(video_id)])
    return str(out)
