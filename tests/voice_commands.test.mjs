// Run: node --test tests/voice_commands.test.mjs  (checks the voice command parser in api/static/index.html, which has no build step)
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const html = readFileSync(new URL("../api/static/index.html", import.meta.url), "utf8");
const src = html.slice(html.indexOf("const NUMS"), html.indexOf("// end parseCommand"));
const parseCommand = new Function(`${src}; return parseCommand;`)();

const cases = {
  "Play the second one": { type: "cite", n: 2 },
  "play S2": { type: "cite", n: 2 },
  "go to source three": { type: "cite", n: 3 },
  "next": { type: "step", by: 1 },
  "play the next one": { type: "step", by: 1 },
  "previous": { type: "step", by: -1 },
  "back one": { type: "step", by: -1 },
  "replay": { type: "step", by: 0 },
  "go back 10 seconds": { type: "seek", by: -10 },
  "rewind": { type: "seek", by: -10 },
  "skip ahead thirty seconds": { type: "seek", by: 30 },
  "go forward a minute": { type: "seek", by: 60 },
  "Pause.": { type: "pause" },
  "play": { type: "resume" },
  "OK vidcite, stop": { type: "pause" },
};
for (const [text, want] of Object.entries(cases)) test(text, () => assert.deepEqual(parseCommand(text), want));

test("questions are not commands", () => {
  for (const q of ["Should I use HNSW for vector search?", "What is the first normal form?", "How do I play a video in Figma?",
                   "What does the second speaker say about indexes?"]) assert.equal(parseCommand(q), null, q);
});
