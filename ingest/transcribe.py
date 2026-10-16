"""Transcribe audio to word-level timestamps.

On Apple Silicon Macs we use mlx-whisper, which runs on the Mac's GPU.
Everywhere else (or if mlx-whisper isn't installed) we fall back to
faster-whisper on CPU. Both produce the same output format.

Results are cached in data/transcripts/<video_id>.json so re-chunking
never requires re-transcribing (the slow step).
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

TRANSCRIPT_DIR = Path("data/transcripts")

# Technical terms Whisper tends to misspell; extend this for your talks.
GLOSSARY = "Kafka, Postgres, pgvector, Kubernetes, Raft, Paxos, HNSW, RAG, LLM, sharding, gRPC"

MLX_MODEL = "mlx-community/whisper-small.en-mlx"   # try whisper-large-v3-turbo for accuracy
CPU_MODEL = "small.en"


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _words_mlx(audio_path: str) -> list[dict]:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=MLX_MODEL,
        language="en",
        word_timestamps=True,
        initial_prompt=GLOSSARY,
    )
    return [
        {"text": w["word"].strip(), "start": round(w["start"], 2), "end": round(w["end"], 2)}
        for seg in result["segments"]
        for w in seg.get("words", [])
        if w["word"].strip()
    ]


def _words_faster_whisper(audio_path: str) -> list[dict]:
    from faster_whisper import WhisperModel

    model = WhisperModel(CPU_MODEL, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        audio_path,
        language="en",
        word_timestamps=True,
        vad_filter=True,          # skip silence and music
        initial_prompt=GLOSSARY,  # nudges spelling of technical terms
    )
    return [
        {"text": w.word.strip(), "start": round(w.start, 2), "end": round(w.end, 2)}
        for seg in segments
        for w in (seg.words or [])
        if w.word.strip()
    ]


def transcribe(video_id: str, audio_path: str, force: bool = False) -> list[dict]:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    cache = TRANSCRIPT_DIR / f"{video_id}.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())

    words = None
    if _is_apple_silicon():
        try:
            words = _words_mlx(audio_path)
        except ImportError:
            print("mlx-whisper not installed; falling back to faster-whisper on CPU")
    if words is None:
        words = _words_faster_whisper(audio_path)

    cache.write_text(json.dumps(words))
    return words
