# First run notes

## Interview summary

Got vidcite (a RAG app that answers questions about Creative Commons YouTube talks with
clickable timestamp citations) running end to end on a MacBook Air M2 for the first time.

**Result.** 2 talks ingested (2,145 words / 19 chunks and 2,042 words / 11 chunks).
"Should I use HNSW for vector search" returns a correct answer citing 3:40 and 4:24,
top_score 0.75. CLI latency went 23.0 s → 8.6 s → ~5.3 s; the LLM call itself is ~0.7 s.

**Stories worth telling:**
1. *Same symptom, three causes.* "Gemini doesn't work" was a retired model (404), real
   free-tier overload (503), and a parameter from the previous model generation
   (`thinking_budget` vs `thinking_level`, 400). Reading the status code first, then the
   docs, separated them. The fix was a different stable model plus the right parameter,
   not retry loops. (Entries 4, 7, 8)
2. *The GPU was 8x slower.* Profiled a 7 s CLI run: one short query on a small embedding
   model takes 111 ms on the Mac GPU and 14 ms on CPU, because transfer/warm-up dominates.
   Moved queries to CPU and kept the GPU for batch ingest, then proved the change was safe
   (CPU and GPU vectors have cosine 1.0) before trusting it. (Entries 9b, 9d)
3. *Designing for a provider swap paid off.* Switching Anthropic → Gemini touched one code
   file, because `api/llm.py` was the only place that talks to the LLM. Kept it that way by
   wrapping provider errors in our own `LLMUnavailable` instead of leaking Gemini types. (Entries 3, 9a)
4. *Tests passed while config was broken.* 30/30 unit tests passed with the wrong API key,
   because the tests inject a fake LLM, which is right for unit tests. It's why a live smoke test
   belongs in the runbook. (Entry 3)
5. *Verify the scary message.* A fresh-cache fallback printed "Creating a new one with mean
   pooling", which would silently corrupt embeddings if true. Checked the pooling mode and vector
   equality instead of assuming either way: it was a misleading log line. (Entry 9d)

**Deliberately deferred:** citation links can be up to ~45 s early (chunk-start granularity),
sponsor/outro chunks are indexed, and captions misspell terms. Each one needs the Phase 3 eval
to measure the fix.

---

## Detailed log

Running vidcite end to end for the first time on a MacBook Air M2 (8 GB RAM, macOS),
against the existing Supabase project "vidcite" (schema already applied).
Each entry: what happened, why, what we changed.

## 1. Missing prerequisites: Python 3.12 and ffmpeg

**What happened.** `python3.12` and `ffmpeg` were not on the PATH. The only
`python3` was 3.10.6 from the python.org installer.

**Why it matters.** The code needs Python 3.11+ (the README says so, and it uses
`str | None` annotations that are evaluated at runtime in dataclasses and
FastAPI/pydantic models). ffmpeg is needed by yt-dlp to extract 16 kHz WAV audio
for the Whisper path (`--source whisper`). The captions path doesn't need it.

**What we changed.** No code changes. Installed both with Homebrew:
`brew install python@3.12 ffmpeg` (Python 3.12.15, ffmpeg 9.0.2). The old
Python 3.10 is left untouched; the venv is created explicitly with
`/opt/homebrew/bin/python3.12`.

## 2. `pip install` failed: disk full (`[Errno 28] No space left on device`)

**What happened.** Creating the venv worked, but `pip install -r requirements.txt`
died partway through installing packages with `OSError: [Errno 28] No space left on device`.

**Why.** The Mac only had about 458 MB free on its 460 GB disk. The requirements
pull in a heavy ML stack (torch, mlx, onnxruntime, ctranslate2, scipy, llvmlite,
transformers) of roughly 2.5 GB installed, and pip also needs room to download
and unpack the wheels. It ran out after writing 1.4 GB into `.venv` and caching
1.1 GB in `~/Library/Caches/pip`. Not a code problem.

**What we changed.** No code changes. Freed disk space outside the project
(free space went from 458 MB to 157 GB), deleted the half-built `.venv`, recreated
it with Python 3.12 and reinstalled. pip reused the wheels it had already cached,
so the retry was quick. Lesson: check `df -h` before installing an ML stack; a full
disk shows up as a confusing mid-install `OSError`, not a clear warning.

## 3. `.env` had a Gemini key instead of an Anthropic key

**What happened.** pytest passed (30/30), but the `.env` check found
`ANTHROPIC_API_KEY` empty. Without printing any values, listing the variable
names showed `.env` held `DATABASE_URL` and `GEMINI_API_KEY`.

**Why it matters.** The answer step talks to Claude through the Anthropic SDK
(`api/llm.py` → `anthropic.Anthropic()`), which only reads `ANTHROPIC_API_KEY`.
A Gemini key wouldn't work even under the right name, since it's a different
provider. The tests didn't catch it because they inject a fake `complete()`, which
is good unit-test design but means config errors only show up on a real run.

**What we changed.** Decided to switch providers rather than get an Anthropic key,
because Gemini has a free tier. Because `api/llm.py` was designed as the single
place that talks to the LLM, the swap stayed small:
- `api/llm.py`: same `complete(user_message) -> str` signature, now calling
  `google-genai`'s `models.generate_content` with the same system prompt,
  `temperature=0` and a 400-token cap. Nothing else in the pipeline changed.
- `requirements.txt`: `anthropic` → `google-genai`.
- `.env.example` and README setup step 6: `ANTHROPIC_API_KEY` → `GEMINI_API_KEY`.
- Turned Gemini "thinking" off (`thinking_budget=0`), because thinking tokens count
  against `max_output_tokens` and could use up the 400-token budget before any answer text.
- `response.text` can be `None` when a response is blocked, so it falls back to `""`;
  `answer.py` then treats it as "not found" (no valid citations) instead of crashing.
- pytest still 30/30 (the tests inject a fake `complete()`).

Side note: the original default model, `claude-haiku-5-5`, wasn't a real Anthropic
model ID, so the Anthropic path would have returned a 404 even with the right key.

## 4. Picking a Gemini model: 404, then 503

**What happened.** The first default, `gemini-2.5-flash`, appears in `models.list()`,
but calling it returned `404 NOT_FOUND ... no longer available to new users`. Switched
to `gemini-3.8-flash`, the model the error message itself recommended. That and three
other Flash models (`gemini-flash-latest`, `gemini-3.5-flash`, `gemini-3.1-flash-lite`)
all returned `503 UNAVAILABLE ... high demand`.

**Why.** Two separate things. (1) A model being listed doesn't mean your key can call
it; Google retires older models for new keys while still listing them. (2) A 503 is
Google's capacity, not our request: the request shape was accepted (no 400 about the
model ID or thinking config), and free-tier traffic is the first to be shed during spikes.

**What we changed.** Default model is now `gemini-3.8-flash` (overridable with `LLM_MODEL`).
No retry/backoff added yet; that would be a design change. Retrying later first.


## 5. First ingest: worked, plus things to watch

**What happened.** `python -m ingest.cli P8rrhZTPEAQ A2QrRXCypm4 --source youtube`
succeeded in ~46 s total, including the first download of the bge-small model.

| Video | Duration | Words | Chunks | Avg chunk |
|---|---|---|---|---|
| P8rrhZTPEAQ "99% of Developers Don't Get PostgreSQL" | 759 s | 2145 | 19 | 46 s |
| A2QrRXCypm4 "Give me 8 minutes and I will make you an AI Engineer" | 461 s | 2042 | 11 | 46 s |

DB row counts matched the JSON files (19 + 11). Every chunk ended on a sentence boundary,
with no gaps and a ~4–9 s one-sentence overlap, which is what the chunker was designed to do.

**Suspicious, not fixed:**
- *Last chunk ends ~2 s after the video does* (761.1 s vs 759 s; 463.3 s vs 461 s). The final
  caption event's `dDurationMs` runs past YouTube's integer duration. Harmless for `?t=` links,
  but an eval that checks "timestamp within video" would flag it.
- *Caption misspellings of technical terms*: "Postgrql", "tpples" (tuples), "Superbase"
  (Supabase). These hurt exact-term matching; the Whisper path's glossary prompt exists for this.
- *A sponsor read is indexed as content* (PostgreSQL video, around 6:18–8:00, plugging Supabase).
  It could get retrieved for questions about realtime/hosting.
- *Tiny outro chunks* ("thanks for watching", 11 s / 37 words) add low-value rows.
- *yt-dlp warning*: "No supported JavaScript runtime could be found" (it wants `deno`). It works
  today, but YouTube extraction without a JS runtime is deprecated and may break later.
  Didn't install deno, to avoid adding a dependency without asking.

## 6. First answer: works, with `gemini-3.7-flash`

**What happened.** `python -m api.ask_cli "Should I use HNSW for vector search"` crashed with
Gemini's `503 high demand` again. Retrieval had already worked by then: the code only calls the
LLM after the confidence gate passes. Running retrieval alone showed top_score 0.750 (the
"Just use HNSW" chunk at 4:24), with all 5 hits from the right video and 59 ms warm search time.
A one-shot probe of each Flash model showed `gemini-3.7-flash` answering, others 503, and two
"lite" models returning `400 INVALID_ARGUMENT` (likely they reject `thinking_budget=0`).

With `LLM_MODEL=gemini-3.7-flash` (env override, no code change) the CLI returned a correct,
cited answer: found=True, top_score=0.75, latency 22,999 ms, citations at 4:24 and 3:40.

**Why it was slow.** Warm retrieval is ~60 ms, so nearly all of the 23 s is the cold load of
the embedding model plus Gemini. The google-genai SDK retries 503s internally (a failing
call once took 29.7 s before raising), so a busy free tier shows up as latency, not an error.

**Suspicious.** Citations point at the *chunk start*: [S2] links to 3:40, but the HNSW-vs-IVFFlat
part of that chunk is later. With ~46 s chunks a link can land up to ~45 s early. Couldn't
verify the NOT_FOUND path live (that probe hit a 503).

**What we changed.** Nothing in code for this step; used the `LLM_MODEL` override.

## 7. Default model changed to `gemini-3.7-flash`, then it got busy too

**What we changed.** One line in `api/llm.py`: default `LLM_MODEL` `gemini-3.8-flash` →
`gemini-3.7-flash`, the model that answered in step 6. pytest still 30/30.

**What happened next.** The very next CLI run with the new default hit `503 high demand`.
A model that worked minutes earlier was busy again.

**Lesson.** On the free tier, *which* model has capacity changes minute to minute, so picking
"the one that works right now" isn't a real fix. The real gap is that a 503 crashes the CLI
with a traceback instead of a clean message. Options (not yet done; they'd be design changes):
a fallback list of models, or catching the error and printing "LLM busy, try again".

## 8. Root cause found: wrong thinking parameter for Gemini 3.x; switched to `gemini-3.5-flash-lite`

**What happened.** Checked Google's docs instead of guessing. The models page recommends
`gemini-3.8-flash` or `gemini-3.5-flash-lite` for new projects (2.5 models are limited to
existing users). The thinking docs say Gemini 3.x uses `thinking_level`
(`minimal`/`low`/`medium`/`high`); `thinking_budget` is the 2.5-era parameter.

**Why.** The `400 INVALID_ARGUMENT` from `gemini-3.5-flash-lite` in step 6 was *our* bug,
not capacity: we sent `thinking_budget=0`, which that model rejects. (`gemini-3.8-flash`
rejects `minimal`; its lowest is `low`.) So the 503s were real overload on the bigger Flash
models, but the lighter stable model had been available all along. We were just calling it wrong.

**What we changed.** `api/llm.py`: default model `gemini-3.5-flash-lite`, and
`ThinkingConfig(thinking_level="minimal")` replacing `thinking_budget=0`. pytest 30/30.
Direct test: correct cited answer in 680 ms, and `NOT_FOUND` for an off-topic question
(first live check of the not-found path). Full CLI: found=True, top_score 0.75,
latency 8,562 ms (down from 22,999 ms).

**Remaining latency.** The LLM call is now ~0.7 s and warm search ~60 ms, so most of the 8.5 s
is loading the bge-small model onto the Mac GPU on the first question of each CLI process.
The FastAPI server already avoids this by loading the model once at startup.

**Interview takeaway.** "Model unavailable" had three different causes that all looked like
"Gemini is broken": a retired model (404), real overload (503), and a wrong parameter for the
model generation (400). Reading the status code first, then the docs, separated them.

## 9. Cleanup pass on the loose ends

### 9a. Busy/failed LLM crashed the CLI with a traceback (fixed)
**Why.** Any Gemini error (503 overload, 404 retired model) propagated as a raw
`google.genai.errors.*` exception through `answer_question` into the CLI.
**What we changed.** `api/llm.py` wraps provider errors in its own `LLMUnavailable`
exception, so callers never import Gemini-specific errors and the "one-file swap" design
still holds. `api/ask_cli.py` catches it, prints a two-line message, exits with code 1,
and closes the DB connection in `finally`. Verified by forcing a 404 with
`LLM_MODEL=gemini-2.5-flash`; normal run unchanged; pytest 30/30.

### 9b. Why the CLI takes ~7 s (measured, not yet changed)
Imports of torch + sentence-transformers ~3.8 s; model load 1.7 s (0.4 s with
`HF_HUB_OFFLINE=1`, so ~1.2 s is an online check against Hugging Face); first encode on the
Mac GPU (MPS) 1.74 s versus 0.04 s on CPU; warm single-query encode 111 ms (MPS) versus 14 ms (CPU).
**Insight.** For one short query on a 33M-parameter model, the GPU is ~8x *slower* than the
CPU: transfer and kernel-launch overhead outweigh the compute. "Use the GPU" isn't automatically
faster. The CLI's `latency_ms` also includes this cold start, so it overstates per-question latency.

### 9c. Citation links land early (measured)
For the HNSW answer, [S2] links to 3:40 (chunk start) but "HNSW" is first said at 4:09,
30 s later. Chunks average ~46 s, so links can be up to ~45 s early. Only ~4 s of that is
the one-sentence overlap; the rest is chunk granularity.

### 9d. Faster question embedding: CPU for queries, skip the online model check (fixed)
**Why.** Measured in 9b: for a single short query, bge-small is ~8x faster on CPU than on
the Mac GPU, and every model load made a ~1.2 s round trip to Hugging Face even though the
model was already on disk.
**What we changed.** `ingest/embed.py`: `get_model(device)` caches one model per device;
`embed_query` uses `QUERY_DEVICE = "cpu"`, while `embed_passages` (ingest batches) still
prefers the Mac GPU. Models load with `local_files_only=True` and fall back to a normal
download if the model isn't cached. `api/main.py` pre-loads the CPU model at startup,
since that's the one questions use.
**Verified.**
- pytest 30/30.
- CLI `latency_ms` 7,105 → ~5,100–5,600 ms; same answer, same top_score 0.75.
- CPU and GPU produce the same query vector (cosine 1.000000), so questions embedded on CPU
  still match chunks embedded on the GPU. This is the check that would catch a silent quality regression.
- Fresh-machine fallback tested with an empty `HF_HOME`: it downloads and loads correctly
  with **CLS pooling** and vectors identical to the normal cache. It does print a scary
  "Creating a new one with mean pooling" line once (from the failed offline attempt);
  that line is misleading, and I verified the pooling mode before trusting it.
**Remaining.** ~3.8 s of each CLI run is just importing torch + sentence-transformers.
Only a long-running process (the FastAPI server) removes that.

### 9e. yt-dlp "No supported JavaScript runtime" warning (fixed)
**Why.** YouTube increasingly requires running its player JavaScript to extract formats;
yt-dlp uses `deno` for that and warned that extraction without it is deprecated.
**What we changed.** `brew install deno` (2.9.7). No code change. Re-ran
`ingest.cli A2QrRXCypm4 --no-store`: 0 warnings, identical output (2042 words, 11 chunks).

### Still open (deliberately deferred to the Phase 3 eval)
- Citation links point at chunk start (up to ~45 s early). Fix needs sentence-level
  timestamps per chunk (schema change + re-ingest).
- Sponsor segment and outro chunks indexed as content; caption typos ("Postgrql").
  Filters should be measured against the eval set, not added blind.
