# Video Q&A with timestamp citations

Ask a question about a library of long technical talks; get a short answer with
clickable timestamps that jump the video to the exact moment.

![Animated walkthrough of the vidcite demo: an answer with citation buttons, the embedded player jumping to each cited moment, and a refusal for a question the talks don't cover](docs/demo.gif)

**[Try the recorded demo →](https://nishithvadlamudi.github.io/vidcite/)** (real answers from an eval run, replayed without a server)

*Walkthrough made from screenshots of the demo page; the answers are real responses recorded during an eval run.*

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
# Web UI + API: open http://localhost:8000 (UI) or http://localhost:8000/docs (API)
uvicorn api.main:app --reload

# Or from the terminal
python -m api.ask_cli "How does Raft elect a leader?"
```

![vidcite local web UI: an answer about why PostgreSQL is object-relational, with citation buttons S1 and S2 and an embedded player paused at the cited moment](docs/screenshot.jpg)

*The local web UI. The answer shown is a real response recorded during an eval run (1.6 s), replayed through
the page's own renderer.*

The web UI (`api/static/index.html`, no build step or dependencies) shows the answer with its `[S1]` tags
as buttons that jump an embedded YouTube player to the cited second, the matching moments, and the talks
in the library. If the LLM is unavailable (e.g. the Gemini free-tier quota), `/ask` falls back to
**search only**: the best matching moments with timestamp links and no generated answer, still behind the
confidence gate.

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
python -m evals.run --llm    # also end-to-end answers, citations and abstention (rate-limited;
                             # ~70-80 Gemini calls, free tier allows 500/day)
```

Results on 2 talks (29 answerable + 8 unanswerable, `gemini-3.5-flash-lite`):

| | Phase 3 | Phase 4 | Phase 4.1 |
|---|---|---|---|
| Retrieval | vector | hybrid (RRF) | hybrid + keyword gate |
| hit@1 / hit@5 / MRR | 59% / 93% / 0.74 | 76% / 100% / 0.87 | 76% / 100% / 0.87 |
| Answered / cited the gold moment | 90% / 86% | 93% / 93% | **100% / 100%** |
| Cited link on time (3 s late to 15 s early) | 37% | 93% | 93% |
| Abstained on unanswerable | 100% | 100% | 100% |
| p50 latency (warm) | 882 ms | 1,163 ms | 1,023 ms |

**Expanded eval (4 talks, 53 answerable + 12 unanswerable):**

| | Hybrid (Phase 4.1) | + cross-encoder reranker | + Whisper for Figma (`source=auto`) | + window (MaxP) reranking |
|---|---|---|---|---|
| hit@1 / hit@5 / MRR | 64% / 94% / 0.78 | 79% / 94% / 0.86 | 79% / 96% / 0.87 | **85% / 98% / 0.90** |
| Answered | 52/53 | 53/53 | 53/53 | 53/53 |
| Cited the gold moment | 50/53 | 50/53 | 51/53 | **52/53** |
| Cited link on time | 39/50 (78%) | 43/50 (86%) | 42/51 (82%) | **45/52 (87%)** |
| Unanswerable refused | 12/12 | 12/12 | 12/12 | 12/12 |

With multiple gold moments allowed where the talk justifies them (2 questions; see notes entry 18):
retrieval hit@5 **100%**, MRR 0.91, sentence links 83% on time.

**6 talks, 93 questions** (77 answerable + 16 unanswerable; two Whisper-only talks added): retrieval hit@1 77%,
hit@5 92%, MRR 0.83. The informal "ChatGPT vs Gemini Pokemon" commentary is by far the hardest talk (hit@5 67%).
End-to-end numbers for the 93-question set are pending: the run hit the Gemini free tier's 500 requests/day limit.

Per talk (before multi-gold): Postgres 17/17 gold citations, AI engineering 12/12, ChatGPT image tutorial 11/12,
Figma 12/12 (Whisper transcript) but only 7/12 links on time: mostly the 8 s lead-in plus answers that
start mid-sentence (tuned on lecture talks; a tutorial-specific value isn't supported by 12 questions yet).

- Retrieval fuses vector search with Postgres full-text search (reciprocal rank fusion;
  `HYBRID=0` turns it off).
- The confidence gate passes a question if the best cosine is at least `MIN_SCORE` *or* one
  chunk contains every content word of the question (recovers short acronym questions like
  "What is an LSN?"). The LLM's NOT_FOUND rule remains the safety net.
- `source=auto` (default): each video uses its Whisper transcript if you ran Whisper on it
  (`--source whisper`), otherwise its YouTube captions. Currently only the Figma tutorial (unpunctuated
  captions) uses Whisper.
- A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, CPU, ~0.6 s per question) reranks the
  top 10 fused candidates, scoring each chunk by its best 3-sentence window so an answer in the middle
  of a mixed-topic chunk isn't diluted (`RERANK=0` disables it; `RERANK_MODEL=BAAI/bge-reranker-base` is ~2 points
  better on hit@1 but ~4x slower).
- Citation links jump to the sentence that best matches the cited claim, estimated from its
  word offset within the chunk, minus an 8 s lead-in tuned on the dev split.
- Measured and **left off**: merging adjacent chunks (`MERGE=1`) removes duplicate links but
  drops on-time links from 93% to 83%. Whisper transcripts (`--source whisper`) tie with YouTube
  captions on this set, so captions stay the default (Whisper is for videos without captions).

Analysis in `docs/first-run-notes.md`.

## Recorded demo (GitHub Pages)

`docs/` is a static demo served by GitHub Pages: `python scripts/build_demo.py` copies 12 real answers
(with citations) from recorded eval runs into `docs/demo-data.json` and generates `docs/index.html` from the
live UI, so the two share styles and rendering code. No server, no API keys, no cost.

## Deploy (Hugging Face Spaces)

Note: Hugging Face now requires a paid plan for Docker Spaces (Static Spaces stay free), so this isn't
deployed; the recorded demo above is the free option. The `Dockerfile` also runs unchanged on other container
hosts (it was tested locally). The `Dockerfile` serves the web UI + API only (CPU torch, both models baked in, port 7860); ingestion
stays local. Public-demo limits are on in the image: 10 questions per visitor per hour and 150 generated
answers per day, after which `/ask` answers in search-only mode (`api/limits.py`).

1. Create a read-only database login: run `deploy/hf-space/readonly_role.sql` in Supabase's SQL editor
   (choose your own password).
2. Create a **Docker** Space (free CPU basic). In its settings add secrets `DATABASE_URL` (the read-only
   login) and `GEMINI_API_KEY` (ideally a key from a separate Google project, so the demo can't use your
   eval quota).
3. Commit, then `scripts/deploy_space.sh https://huggingface.co/spaces/<user>/vidcite`. It builds the Space
   from `HEAD` only, so `.env` and uncommitted files never leave your machine.

## Status

- [x] Schema
- [x] License-checked metadata, captions and audio download
- [x] Two transcript sources: YouTube captions (json3) and Whisper
- [x] Timestamp-aware chunker + tests
- [x] Ingestion CLI
- [x] Embedding (bge-small) + storage in Supabase
- [x] Phase 2: vector search, confidence gate, citation-tagged prompt, `/ask` API + CLI
- [x] Phase 3: eval set and metrics (`evals/`)
- [x] Phase 4: keyword search + rank fusion, sentence-level citation links
- [x] Phase 4.1: keyword gate, neighbour merging (measured, off), Whisper vs captions eval
- [x] Eval expanded to 4 talks / 65 questions; chunker fallback for unpunctuated captions
- [x] Cross-encoder reranking (on by default)
- [x] `source=auto`: Whisper where it was run (Figma), captions elsewhere
- [x] Local web UI with embedded player; search-only fallback when the LLM is unavailable
- [x] Two talks that YouTube blocked for captions ingested via Whisper (6 talks)
- [x] Gold questions for the two new talks (93-question eval)
- [x] Deployment files for Hugging Face Spaces, usage limits, read-only DB role SQL, CC BY credits in the UI
- [ ] Next: create the Space; 93-question end-to-end run (after the quota resets)
