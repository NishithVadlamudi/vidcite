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

6. *The eval overturned a tuning decision.* Tuning the similarity cutoff on dev data said 0.61;
   the holdout showed no gain, and the end-to-end run showed the LLM's NOT_FOUND rule, not the
   cutoff, rejected 6 of 8 unanswerable questions, including near-topic ones that outscored real
   answers. Kept 0.55. Results: hit@5 93%, 86% gold citations, 100% abstention. (Entry 10)

7. *Two measured fixes, each checked on holdout.* Hybrid search (pgvector + Postgres full-text,
   fused with RRF) took hit@1 from 59% to 76% and hit@5 to 100%, fixing rare terms like
   "postmaster" and "LSN". Sentence-level citation links (no schema change: sentence times
   estimated from word offsets) plus an 8 s lead-in tuned on dev took on-time links from
   37% to 93% end to end. (Entry 11)

8. *Reporting a negative result.* Merging adjacent chunks looked like an obvious win (hit@1 76% -> 93%),
   but that number was inflated by longer passages, and end to end it made citation links worse
   (93% -> 83% on time) with no gain in answers. Kept it off and fixed the visible symptom (duplicate
   links) in the CLI instead. Similarly, Whisper tied with free captions, so captions stay the default. (Entry 12)

**Deliberately deferred:** sponsor/outro chunks are indexed; "summary" chunks outrank specific
moments (seen on the ChatGPT tutorial); two talks still blocked by YouTube 429s. Each fix should be measured with the eval.

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

## 10. Phase 3: eval set and metrics

**What we built (no new dependencies).**
- `evals/questions.jsonl`: 29 answerable questions (17 Postgres talk, 12 AI talk) and 8
  unanswerable ones, 5 of which are deliberately near-topic (MySQL locking, Postgres
  partitioning, Kafka partitions, BM25, GPT-5 pricing). Gold timestamps aren't eyeballed:
  each question has a short gold phrase, and the second it's spoken is looked up in the
  caption word timings. Every third question is held out (20+5 dev, 9+3 holdout).
- `evals/metrics.py`: pure functions (hit@k, MRR, link error, gate pass/reject rates,
  threshold picker); 6 new unit tests, 36 total.
- `evals/run.py`: retrieval eval with no LLM calls, a MIN_SCORE sweep tuned on dev and
  reported on holdout, and `--llm` for end to end with a 7 s delay for free-tier limits.
  Raw results go to `data/evals/` (gitignored) so runs can be compared.

**Results (youtube captions, k=5, gemini-3.5-flash-lite).**

| Metric | dev | holdout | all |
|---|---|---|---|
| hit@1 | 60% | 56% | 59% |
| hit@5 | 95% | 89% | 93% |
| MRR | 0.75 | 0.70 | 0.74 |
| citation link <=15 s early | 53% | 12% | 41% |

End to end: 90% of answerable questions answered, 86% cited the gold moment, **100% of
unanswerable questions abstained**, p50 latency 882 ms, 0 LLM errors.

**What the numbers say.**
1. *The score gate can't detect near-topic questions; the LLM can.* Unanswerable top scores
   reach 0.740 (Postgres partitioning) and 0.705 (MySQL locking), higher than answerable ones
   like "What is an LSN?" (0.516). Tuning MIN_SCORE on dev picked 0.61 (98% balanced on dev)
   but it scored 67% on holdout, the same as the current 0.55: textbook overfitting, caught
   by the holdout. In the end-to-end run the gate stopped 2 of 8 unanswerable questions and
   the prompt's NOT_FOUND rule stopped the other 6. So the gate is a cost filter, not the
   safety mechanism, and MIN_SCORE was left at 0.55.
2. *The gate costs recall.* "What is an LSN?" (0.516) was rejected before the LLM saw the right
   chunk (rank 3). Short acronym questions embed weakly.
3. *Caption typos hurt retrieval, measurably.* "What metadata does each tuple carry?" ranked a
   chunk from the *other* video first, because the captions say "tpples", never "tuple".
4. *Rare terms miss entirely.* "What is the postmaster?" has the gold chunk outside the top 5
   (top 1 was the intro). Keyword search over the existing `tsv` column (Phase 4) targets this.
5. *Many hit@1 misses are the neighbouring chunk.* E.g. "object-relational": gold at 1:19, top 1
   starts at 1:24 with the explanation sentence. The strict point-in-span metric undercounts
   near misses; MRR and hit@5 tell the fuller story.
6. *Citation links are the weakest part.* Only 41% land within 15 s (median 17 s early, max 44 s).
   That puts a number on the chunk-start issue from 9c and makes sentence-level links the
   top Phase 4 item.
7. *The lite model is slightly over-cautious.* "How does a typical RAG pipeline prepare documents?"
   got NOT_FOUND although the gold chunk was retrieved (rank 3).

**Caveats.** 37 questions over 2 talks, written by the same person tuning the system. The holdout
has only 3 unanswerable questions, so its gate numbers move in 33-point steps. Good for spotting
failure modes, not for claiming a precise accuracy.

**Interview takeaway.** The eval changed a decision: without it, I'd have "tuned" MIN_SCORE to 0.61
from dev numbers. The holdout showed it didn't generalise, and the end-to-end run showed the
LLM's abstention, not the threshold, is what keeps answers honest.

## 11. Phase 4: hybrid search and sentence-level citation links

**A. Keyword search + reciprocal rank fusion.** `retrieval/search.py` adds a Postgres
full-text query over the existing `tsv` column and GIN index (no schema change), using any-word
matching (`plainto_tsquery` ANDs the words, so the `&` is swapped for `|`) ranked by
`ts_rank_cd`. Top 20 from vector and top 20 from keyword are fused with RRF (k=60). Passages
keep their cosine score, so the confidence gate works unchanged; `top_score` is now the max
cosine, because the first fused passage isn't always the closest.
- *Bug on the way:* the first SQL put `...::tsquery as query` straight into `FROM`; Postgres only
  allows function calls there, not casts. Fixed by wrapping it in a one-row subquery.
- *Result (retrieval eval, all 29):* hit@1 59% -> 76%, hit@5 93% -> 100%, MRR 0.74 -> 0.87,
  improving on both dev and holdout. "postmaster" went from not retrieved to rank 2; "LSN" from
  rank 3 to 1; the "tpples" typo question from the wrong video to rank 1 (via the exact terms
  xmin/xmax). Made the default (`HYBRID=0` to disable).
- *Not fixed by it:* "What is an LSN?" still has max cosine 0.516, so the gate rejects it even
  though the right chunk is now rank 1. A gate that also trusts strong keyword matches is a candidate.

**B. Sentence-level citation links, without re-ingesting.** Chunks store only start/end times,
so `retrieval/moment.py` estimates each sentence's start from its word offset (assuming an even
speaking rate within a ~45 s chunk). For each citation, the answer sentences that cite it are
embedded (CPU) and matched against the chunk's sentences; `link_s` is that sentence's start.
`start_s`/`end_s` still describe the chunk; the CLI and API use `link_s` for `?t=`.
- *Metric change:* "on time" became 3 s late to 15 s early. Chunk-start links could only be
  early, but sentence links can be late, and late is worse (the viewer misses the start).
- *First result:* median distance from gold 17 s -> 3 s, on time 41% -> 69%, but most misses
  were *late* (-3 to -19 s).
- *Lead-in tuned like MIN_SCORE:* swept 0-12 s on dev; 8-10 s tied best (85%); picked 8 s
  (the smallest, so the least unrelated speech). Holdout: 67% -> 89%. It falls off a cliff past
  10 s on both splits, a sign the window is real rather than noise.
- *Bug on the way:* `claims_for` left "Use HNSW ." after stripping tags; reused the same tidy-up
  regex as `resolve_citations`. A unit test with wrong arithmetic (14 words, not 15) also failed
  first; the code was right.

**End to end (`python -m evals.run --llm`, hybrid):**

| | Phase 3 | Phase 4 |
|---|---|---|
| answerable answered | 90% | 93% |
| cited the gold moment | 86% | 93% |
| cited links on time | 37% (chunk start) | 93% (sentence) |
| unanswerable abstained | 100% | 100% |
| p50 latency | 882 ms | 1,163 ms |

The latency cost (~0.3 s) is the extra keyword query plus embedding a chunk's sentences per citation.

**Suspicious / next.**
- Overlapping chunks can produce two citations with the same link (the HNSW answer cites
  [S1] and [S2], both at 4:16). Neighbour merging would dedupe them.
- For the HNSW question the link (4:16) is right for the cited claim (the HNSW vs IVFFlat
  comparison) but ~6 s after the question's gold moment (4:09, where HNSW is introduced).
  Gold labels mark where a topic *starts*; a claim can be later. The metric is a proxy.
- 47 tests pass (11 new: RRF, moments, claims, lead-in clamp, link_s on answers).

## 12. Phase 4.1: keyword gate, neighbour merging, Whisper vs captions

**A. Keyword exception to the confidence gate (kept, on).** The keyword query now also returns
`all_terms` (`tsv @@ plainto_tsquery(question)`: the chunk contains every content word).
`retrieve()` passes the gate if the best cosine is at least MIN_SCORE *or* any retrieved chunk
has all_terms. RRF keeps the flag when the same chunk comes from both lists.
- Retrieval eval: 9 answerable questions get a keyword pass, **0 unanswerable**. Gate pass rate
  on dev went 95% -> 100% with no change to rejections.
- End to end (merge off): answered 90-93% -> **100%**, cited the gold moment 93% -> **100%**,
  abstention still **100%**, links on time 93%, p50 1,023 ms. "What is an LSN?" (cosine 0.516)
  is now answered with a link 11 s before the gold moment.

**B. Neighbour merging (built, measured, left off).** Adjacent retrieved chunks from the same video
are joined into one passage, dropping the overlap sentence.
- Retrieval hit@1 76% -> 93%, but that's mostly an artefact: merged passages are longer (3.2 passages
  per question instead of 5, about the same 220-230 s of talk), so "first passage contains gold" is
  easier. hit@5 unchanged at 100%.
- Sentence links got *worse*: 86% -> 62% on time. First hypothesis: estimating sentence times over a
  90-140 s merged span drifts more. Fixed that by estimating per original chunk and carrying the times
  on the merged passage: only 62% -> 66%. So the real cause was sentence *selection*: more candidate
  sentences means more chances to pick one elsewhere in the passage.
- End to end: identical answer quality (100% / 100% / 100%), links 93% (off) vs 83% (on).
  **Decision: MERGE stays off** (`MERGE=1` enables it). The symptom it would fix, duplicate links
  (3/29 answers cite two overlapping chunks at the same moment), is handled in the CLI by printing
  each link once with all its tags, e.g. `[S1][S2] ... @ 11:39`. The API still returns every citation.

**C. Whisper vs YouTube captions.** `python -m ingest.cli P8rrhZTPEAQ A2QrRXCypm4 --source whisper`
took **151 s** for ~20 min of video, including audio and model downloads (mlx-whisper small.en on the
M2 GPU, roughly 8x real time). My earlier estimate of 1-2 min per video minute was far too pessimistic.
- Whisper spells "tuple" correctly (7x; captions had "tpples").
- Retrieval: hit@1 79% vs 76%, hit@5 100% both, MRR 0.87 both, sentence links 83% vs 86%. A tie at this
  sample size (1-2 questions). Hybrid search had already rescued the "tpples" question via "xmin"/"xmax".
  Whisper's top scores run slightly higher across the board (0.525-0.889 vs 0.516-0.869), consistent
  with cleaner text.
- **Decision: captions stay the default** (free, instant); Whisper is for videos without captions.

52 tests pass (6 new: merging, all_terms through RRF, keyword gate, per-chunk times, gate metric).

## 13. Adding four more talks: YouTube 429s and unpunctuated captions

**What happened.** `ingest.cli` with four new IDs ingested the first (Figma UI tutorial), then crashed
on the second with `HTTP Error 429: Too Many Requests` while downloading captions. Videos 3 and 4 were
never attempted.

**Why.** Two separate problems. (1) YouTube rate-limits the caption (timedtext) endpoint per machine,
and this machine had made many requests that day. (2) The CLI only treated `PermissionError` (not CC)
and `FileNotFoundError` (no captions) as per-video failures; any other yt-dlp `DownloadError`
propagated and aborted the whole batch.

**What we changed.** `ingest/cli.py` also catches yt-dlp's `DownloadError` and prints `SKIPPED <id>`,
so one bad video doesn't abort the batch. Retried the rest one at a time, 60 s apart: the ChatGPT image
tutorial went in, and two still got 429 (being retried again after a 15-minute wait). pytest 52/52.

**Suspicious: the Figma captions have no punctuation.** Only 2 of 4,212 words end in `.?!`, so
the chunker's sentence splitter falls back to pauses (>= 0.6 s), and this speaker rarely pauses.
Most "sentences" are forced cuts at the 75 s cap: chunks average 64 s, are cut mid-sentence, get no
overlap (the overlap sentence is longer than the 45 s target, so it's dropped), and there's a 1 s fragment.
Sentence-level citation links can't work either: one "sentence" per chunk means links fall back to the chunk start.

**Decision: chunker fallback for unpunctuated transcripts (option A).** Compared against Whisper for
this video only (Figma would then only be searchable under `source=whisper`, because the app queries one
source at a time) and Whisper for every talk (about 10 min of transcription plus six YouTube audio
downloads while being rate-limited). A was the only option that kept Figma answerable without changing
the default source or downloading more from YouTube, and it fixes any future unpunctuated video.

**A finding while simulating it: captions carry no pause information.** `parse_json3` ends each word
where the next begins, so the median gap in the Figma captions is 0.00 s (8 of 4,212 gaps are >= 0.15 s).
The chunker's "long pause = boundary" rule only ever fires on Whisper transcripts. So for captions,
"cut at the longest pause" just means "cut by time".

**What we changed.**
- `ingest/chunk.py`: `is_punctuated()` (sentence-end marks per word; punctuated transcripts run
  0.066-0.092, Figma 0.000, threshold 0.02). Unpunctuated transcripts use `split_pseudo_sentences()`:
  cut at the longest pause 5-15 s in (earliest on ties, so about every 5 s without pause info).
  Punctuated transcripts take exactly the old path. 4 new tests.
- `retrieval/moment.py`: chunk text with no sentence breaks is split into 15-word blocks (~5 s) for
  link estimation. Inserting fake punctuation into the stored transcript was rejected. 1 new test.
- Re-ingested only Figma: 23 chunks (1/64/75 s, no overlap) -> 36 chunks (23/47/54 s, 6.4 s overlap),
  link points about every 4.7 s. 57 tests pass.
- Known limit: cuts are time-based, so chunks still start and end mid-sentence for this kind of video.
  Whisper (`--source whisper`) is the upgrade path if precise boundaries matter.

**Regression check with 4 talks indexed** (original 37 questions): hit@1 76% -> 72%, hit@5 100%,
MRR 0.87 -> 0.86, sentence links 86% on time. The top result still comes from the right talk every
time; the shifts are between neighbouring chunks of the same talk (keyword candidates change with corpus size).

**The two remaining videos (`eI0NM5hM-4o`, `48b5kLp36T8`) were left out.** Caption downloads got 429
three times over ~40 minutes, including after a 15-minute wait, so the block lasts hours, not minutes.
Decided to continue with 4 talks rather than keep hitting YouTube (or switch those two to Whisper, which
would also need YouTube audio downloads and would make them searchable only under `source=whisper`).
They can be added later with the normal `python -m ingest.cli <id> --source youtube`. Nothing partial
was stored: a 429 happens before chunking, and storage is one transaction per video.

**Expanded eval: 65 questions over 4 talks.** I drafted 28 new questions (12 per new talk + 4
near-topic unanswerable) with gold phrases located in the caption timings; the user reviewed them.
Existing questions kept their splits; new ones continue the every-third-is-holdout rule (44 dev, 21 holdout).

*Retrieval* (53 answerable): hit@1 64%, hit@5 94%, MRR 0.78, sentence links on time 82%. The original
29 were unchanged (72% / 100% / 0.86 / 86%); the drop is all in the new talks:
- ChatGPT image tutorial: hit@1 42%, hit@5 83%, MRR 0.58. Its wrap-up chunk (12:04-12:50) restates
  most of the talk's themes and ranks first for 4 different questions. One real wording gap ("my exact
  wording" vs "put the exact text ... inside quotation marks"). One near miss is a metric artefact: gold
  at 7:47 sits 5 s before the neighbouring chunk that is ranked first.
- Figma: hit@5 92%, but sentence links only 55% on time (time-based cuts, as expected).

*End to end*: the first `--llm` run lost **16 of 65 calls to Gemini 503 "high demand"** (spread across
the run; 0 errors earlier the same day with the same model). Re-ran only the failed rows with a
12 s gap (14 recovered, then the last 2), merging into the same results file:

| Talk | Answered | Cited gold | Links on time |
|---|---|---|---|
| Postgres | 17/17 | 17/17 | 15/17 |
| AI engineering | 12/12 | 12/12 | 10/12 |
| ChatGPT images | 11/12 | 10/12 | 9/10 |
| Figma | 12/12 | 11/12 | 5/11 |
| All | 52/53 (98%) | 50/53 (94%) | 39/50 (78%) |

Unanswerable refused 12/12, including the 4 new near-topic ones (Figma auto layout, Figma -> React, image
resolution, animated GIFs) and the old "GPT-5 per-token pricing", which now sits next to a talk that
*does* quote ChatGPT Plus prices. Holdout: 14/16 cited gold vs 34/35 on dev, which is small enough to be noise.
p50 latency was 4.1 s, inflated by the SDK's internal 503 retries during the busy period (1.0 s on a quiet run).

The 3 misses match the retrieval misses: "exact wording" (NOT_FOUND: right chunk never retrieved),
"change part of an image" (cited the neighbouring chunk), "distance between elements in Figma"
(wrong chunk; the answer sits mid-way through an unpunctuated chunk).

**Lesson.** A bigger, more varied eval found a failure mode the first 37 questions couldn't: a
"summary" chunk that restates the whole talk outranks the specific moment. And a provider
outage can silently cut a quarter of an eval run, so the eval counts LLM errors separately instead of
scoring them as wrong answers. That made "retry only the failures" possible.
