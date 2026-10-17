"""Build the static demo (docs/index.html + docs/demo-data.json) for GitHub Pages.

    python scripts/build_demo.py

The demo replays real answers that vidcite produced during end-to-end eval runs, so it
needs no server. Nothing is written by hand: each answer, its citations and its latency
are copied from a results file in data/evals/, with the run it came from recorded next
to it. The page is generated from the live UI (api/static/index.html), so both share
the same styles and rendering code; only the question box and API calls are swapped
for the recorded questions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Valid end-to-end runs with the current settings (hybrid search, MiniLM window reranking),
# newest first. A question's answer comes from the first run where it succeeded.
RUNS = [
    "data/evals/20261016-171945-auto-hybrid-mergeoff-rerankms-marco-MiniLM-L-6-v2.json",
    "data/evals/20261016-130036-auto-hybrid-mergeoff-rerankms-marco-MiniLM-L-6-v2.json",
]
QUESTIONS = [
    "Why is Postgres described as object-relational rather than just relational?",
    "What is TOAST in PostgreSQL?",
    "What is an LSN?",
    "What is the difference between HNSW and IVFFlat indexes?",
    "How do you define a tool for an LLM?",
    "What is reciprocal rank fusion?",
    "Why is ChatGPT better than other AI tools at putting text in images?",
    "How do I get a transparent background for a logo?",
    "How do I draw a perfect circle in Figma?",
    "How do I make an editable copy of someone else's Figma file?",
    "How does MySQL InnoDB implement row-level locking?",  # near-topic, should be refused
    "What is the capital of France?",                       # off-topic, should be refused
]
OUT = Path("docs/demo-data.json")
PAGE = Path("docs/index.html")
LIVE_UI = Path("api/static/index.html")
REPO = "https://github.com/NishithVadlamudi/vidcite"
GROUPS = [("PostgreSQL internals", "P8rrhZTPEAQ"), ("AI engineering", "A2QrRXCypm4"),
          ("ChatGPT image tutorial", "EqEdnI3B23g"), ("Figma tutorial", "FTFaQWZBqQ8"), ("Not in the talks", None)]


def recorded(question: str) -> tuple[dict, str]:
    for path in RUNS:
        for row in json.load(open(path))["rows"]:
            llm = row.get("llm", {})
            if row["question"] == question and llm and "error" not in llm:
                ok = llm["cited_gold"] if row["video_id"] else not llm["found"]
                if ok:
                    return {"answer": llm["answer"], "citations": llm["citations"], "found": llm["found"],
                            "top_score": round(row["top_score"], 3), "latency_ms": llm["latency_ms"],
                            "search_only": False}, Path(path).name[:15]
    raise SystemExit(f"no valid recorded answer for {question!r}")


def build_page() -> str:
    """docs/index.html from the live UI: banner + recorded questions instead of the ask box."""
    s = LIVE_UI.read_text()
    reps = [
        ('<title>vidcite</title>', '<title>vidcite demo</title>\n<meta name="description" content="Recorded demo of '
         'vidcite: answers about Creative Commons tech talks, cited to the second.">'),
        ('  .error { color: #dc2626; }', '''  .error { color: #dc2626; }
  .banner { max-width: 1180px; margin: 0 auto; padding: 0 16px; }
  .banner div { background: var(--accent-soft); border: 1px solid var(--line); border-radius: 10px; padding: 10px 14px; font-size: 14px; }
  .banner a, footer a { color: var(--accent); }
  .qs { display: grid; gap: 6px; }
  .qs button { text-align: left; font-size: 14px; }
  .qs button.active { border-color: var(--accent); color: var(--accent); }
  .qs .group { font-size: 12px; color: var(--muted); margin: 8px 0 0; text-transform: uppercase; letter-spacing: .05em; }'''),
        ("</header>", f'''</header>
<div class="banner"><div><strong>Recorded demo.</strong> These are real answers vidcite produced during an evaluation run,
  replayed here so the demo runs without a server. Pick a question; each citation jumps the player to the cited second.
  For live questions, <a href="{REPO}#setup">run it locally</a> · <a href="{REPO}">source on GitHub</a>.</div></div>'''),
        ('''    <form id="ask">
      <input id="q" type="text" placeholder="e.g. Should I use HNSW for vector search?" autocomplete="off" required minlength="3" maxlength="500">
      <button class="primary" id="go" type="submit">Ask</button>
    </form>
    <div class="examples" id="examples"></div>''', '''    <h2>Pick a question</h2>
    <div class="qs" id="questions"></div>'''),
        # The answer goes above the 12 questions; below them it was off-screen on a laptop.
        ('''    <div id="result" aria-live="polite"></div>''', ""),
        ('''    <h2>Pick a question</h2>''', '''    <div id="result" aria-live="polite"></div>
    <h2 style="margin-top:18px">Pick a question</h2>'''),
        ("credit belongs to the channels listed.</footer>", f'''credit belongs to the channels listed.
  <a href="{REPO}">vidcite on GitHub</a> · <a href="{REPO}/blob/main/docs/first-run-notes.md">build and eval notes</a></footer>'''),
    ]
    for old, new in reps:
        if old not in s:
            raise SystemExit(f"live UI changed; update build_page(): {old[:60]!r}")
        s = s.replace(old, new)
    head, script = s.split("<script>", 1)
    # fmt, el, play, answerNode and render are shared with the live UI; ask() and the
    # /videos fetch after them are live-only.
    shared = script[script.index("const $ = "):script.index("async function ask(question)")]
    live_meta = '''`${r.search_only ? "search only" : r.found ? "answered" : "not found"} · top score ${r.top_score ?? "–"} · ${r.latency_ms} ms`'''
    if live_meta not in shared:
        raise SystemExit("live UI changed; update the meta line replacement in build_page()")
    # Latency is left out: some recorded calls include Gemini retries during a busy period.
    shared = shared.replace(live_meta, '''`recorded answer · ${r.found ? "answered" : "refused: not in the talks"} · top score ${r.top_score ?? "–"}`''')
    demo_js = """
const GROUPS = %s;
fetch("demo-data.json").then((r) => r.json()).then((data) => {
  const videoOf = (a) => a.response.citations[0]?.video_id ?? null;
  const buttons = [];
  for (const [label, vid] of GROUPS) {
    const items = data.answers.filter((a) => videoOf(a) === vid);
    if (!items.length) continue;
    $("questions").append(el("p", { className: "group", textContent: label }));
    for (const a of items) {
      const b = el("button", { type: "button", textContent: a.question, onclick: () => {
        buttons.forEach((x) => x.classList.toggle("active", x === b));
        render(a.response);
        $("result").scrollIntoView({ block: "nearest", behavior: "smooth" });  // phones: answer may be above the fold
      } });
      buttons.push(b); $("questions").append(b);
    }
  }
  $("library").replaceChildren(...data.library.map((v) => el("li", {},
    el("span", { textContent: v.channel ? `${v.title} · ${v.channel}` : v.title }),
    el("span", { textContent: v.duration_s ? fmt(v.duration_s) : "" }))));
}).catch(() => $("questions").replaceChildren(el("p", { className: "error", textContent: "Couldn't load the recorded answers." })));
</script>
</body>
</html>
""" % json.dumps([list(g) for g in GROUPS])
    return head + "<script>\n" + shared + demo_js


def main() -> None:
    from ingest.store import connect

    answers = []
    for q in QUESTIONS:
        response, run = recorded(q)
        answers.append({"question": q, "run": run, "response": response})
    used = {c["video_id"] for a in answers for c in a["response"]["citations"]}
    conn = connect()
    try:
        rows = conn.execute("select id, title, channel, duration_s from videos where id = any(%s) order by title",
                            (sorted(used),)).fetchall()
    finally:
        conn.close()
    library = [{"id": i, "title": t, "channel": c, "duration_s": d} for i, t, c, d in rows]
    OUT.write_text(json.dumps({"answers": answers, "library": library}, indent=1, ensure_ascii=False))
    print(f"wrote {OUT}: {len(answers)} answers, {len(library)} talks")
    PAGE.write_text(build_page())
    (PAGE.parent / ".nojekyll").write_text("")  # serve files as-is on GitHub Pages
    print(f"wrote {PAGE} (from {LIVE_UI})")
    for a in answers:
        r = a["response"]
        print(f"  {a['run']}  {r['latency_ms']:>5} ms  {'answered' if r['found'] else 'refused':8}  {a['question']}")


if __name__ == "__main__":
    main()
