# Video Q&A with timestamp citations

Ask a question about a library of long technical talks; get a short answer with
clickable timestamps that jump the video to the exact moment.

## Setup

1. Install Python 3.11+ and ffmpeg (`brew install ffmpeg`).
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. In Supabase, open the SQL editor and run `db/schema.sql`.
5. Copy `.env.example` to `.env` and paste your Supabase connection string into `DATABASE_URL`.
6. Add your `GEMINI_API_KEY` to `.env` (free key from Google AI Studio).
7. `pytest` (all 30 tests should pass).

## Ingest videos

Find talks with YouTube's search filter: Filters > Features > Creative Commons.
Non-Creative-Commons videos are refused automatically.

```bash
# Fast path: YouTube captions, no transcription (minutes for a whole library)
python -m ingest.cli VIDEO_ID_1 VIDEO_ID_2 --source youtube

# Quality path: Whisper on the Mac GPU via mlx-whisper (slower, cached)
python -m ingest.cli VIDEO_ID_1 --source whisper
```

Each run saves chunks to `data/chunks/<video_id>.<source>.json`, embeds them on the
Mac GPU and stores them in Supabase. Both sources are kept side by side so the eval
can compare them. Add `--no-store` to stop after chunking (no database needed).

## Ask questions

```bash
# From the terminal
python -m api.ask_cli "How does Raft elect a leader?"

# Or run the API, then try it at http://localhost:8000/docs
uvicorn api.main:app --reload
```

Each answer cites its sources as `[S1]`, `[S2]`; the CLI prints a YouTube link that
starts at the cited second. Timestamps come from the database, never from the LLM.
If the best match is too weak, or the model finds no answer, it says so instead of guessing.

## Evaluate

`evals/questions.jsonl` holds hand-labelled questions: answerable ones carry the video
and the second the answer is spoken (looked up from caption word timings via a gold
phrase), and unanswerable ones (several deliberately near-topic) test abstention.
Every third question is a holdout: tune on dev, report on holdout.

```bash
python -m evals.run          # retrieval hit@1/hit@5/MRR, link accuracy, MIN_SCORE sweep (no LLM calls)
python -m evals.run --llm    # also end-to-end answers, citations and abstention (rate-limited)
```

First results (2 talks, 29 answerable + 8 unanswerable, `gemini-3.5-flash-lite`):
hit@5 93%, hit@1 59%, MRR 0.74; end to end 90% answered, 86% cited the gold moment,
100% abstention on unanswerable; but only 40% of citation links land within 15 s
of the answer. Details and analysis in `docs/first-run-notes.md`.

## Status

- [x] Schema
- [x] License-checked metadata, captions and audio download
- [x] Two transcript sources: YouTube captions (json3) and Whisper
- [x] Timestamp-aware chunker + tests
- [x] Ingestion CLI
- [x] Embedding (bge-small) + storage in Supabase
- [x] Phase 2: vector search, confidence gate, citation-tagged prompt, `/ask` API + CLI
- [x] Phase 3: eval set and metrics (`evals/`)
- [ ] Phase 4: keyword search + rank fusion, sentence-level citation links
