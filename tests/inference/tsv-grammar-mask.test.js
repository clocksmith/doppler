import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { createTsvGrammarMask } from "../../src/inference/pipelines/structured/tsv-grammar-mask.js";

/**
 * Unit tests for the TSV grammar mask.
 *
 * The mask enforces line/tab structure at the BPE token-piece level —
 * disallowing newline before all field-separator tabs are emitted, and
 * disallowing tokens that would push past the per-line tab budget.
 * It deliberately does NOT enforce field content (the parser handles
 * that). Tests below pin both halves: the structural constraints fire
 * when expected, and content constraints don't fire spuriously.
 *
 * The fixtures use a tiny synthetic tokenizer where each "token id" is
 * just an index into a string array — pieces map 1:1 to ids. That's
 * enough to exercise the state machine without hooking a real BPE.
 */

function buildSyntheticTokenizer(pieces) {
  return {
    decode(ids) {
      return ids.map((id) => pieces[id] ?? "").join("");
    },
  };
}

function runMask({ pieces, generatedIds }) {
  const tokenizer = buildSyntheticTokenizer(pieces);
  const mask = createTsvGrammarMask({ tokenizer, fieldsPerLine: 4 });
  const logits = new Float32Array(pieces.length).fill(0);
  mask(logits, { generatedIds, vocabSize: pieces.length });
  // Return a map of tokenId → "allowed" | "blocked" for assertions.
  const result = {};
  for (let id = 0; id < pieces.length; id += 1) {
    result[pieces[id]] = logits[id] === -Infinity ? "blocked" : "allowed";
  }
  return result;
}

describe("createTsvGrammarMask", () => {
  it("blocks newline before any tabs are emitted (line cannot end without fields)", () => {
    const pieces = [
      "person_name", // 0 — content piece
      "\t",          // 1 — tab
      "\n",          // 2 — newline (should be blocked)
      "Jane",        // 3 — content
    ];
    const result = runMask({ pieces, generatedIds: [] });
    assert.equal(result["\n"], "blocked", "newline must be blocked at line start");
    assert.equal(result["\t"], "allowed");
    assert.equal(result["person_name"], "allowed");
    assert.equal(result["Jane"], "allowed");
  });

  it("blocks newline mid-line (after some but not all field separators)", () => {
    // Sequence so far: "person_name\tJane Smith" — 1 tab in.
    const pieces = ["person_name", "\t", "Jane Smith", "\n", "0.92", "1"];
    // generatedIds emits "person_name" then "\t" then "Jane Smith"
    const result = runMask({ pieces, generatedIds: [0, 1, 2] });
    assert.equal(result["\n"], "blocked", "newline after only 1 tab must be blocked (need 3)");
    assert.equal(result["\t"], "allowed", "next tab still allowed (advances to confidence)");
  });

  it("allows newline once all 3 tabs are emitted", () => {
    // Sequence: "person_name\tJane\t0.92\t1" — all 3 tabs in.
    const pieces = ["person_name", "\t", "Jane", "0.92", "1", "\n"];
    const result = runMask({ pieces, generatedIds: [0, 1, 2, 1, 3, 1, 4] });
    assert.equal(result["\n"], "allowed", "newline allowed after 3 tabs");
  });

  it("blocks a 4th tab on the same line (would push past the schema)", () => {
    // Sequence: "person_name\tJane\t0.92\t1" — already 3 tabs.
    const pieces = ["person_name", "\t", "Jane", "0.92", "1", "\n", "extra"];
    const result = runMask({ pieces, generatedIds: [0, 1, 2, 1, 3, 1, 4] });
    assert.equal(result["\t"], "blocked", "4th tab on the same line must be blocked");
    assert.equal(result["\n"], "allowed");
  });

  it("resets per-line tab counter on newline", () => {
    // First line completed; cursor is at start of line 2.
    const pieces = ["person_name", "\t", "Jane", "0.92", "1", "\n"];
    const result = runMask({ pieces, generatedIds: [0, 1, 2, 1, 3, 1, 4, 5] });
    // Line 2 is empty so far → newline should be blocked again, tab allowed.
    assert.equal(result["\n"], "blocked", "newline blocked at line 2 start (counter reset)");
    assert.equal(result["\t"], "allowed");
  });

  it("does NOT enforce category content (any string token allowed at line start)", () => {
    // Even nonsense category strings pass through — the parser handles
    // semantic validation. The mask is structural only.
    const pieces = ["definitely_not_a_real_category", "\t", "x", "0.5", "1", "\n"];
    const result = runMask({ pieces, generatedIds: [] });
    assert.equal(result["definitely_not_a_real_category"], "allowed");
    assert.equal(result["x"], "allowed");
  });

  it("handles a multi-character token piece containing tab + newline together", () => {
    // BPE tokens occasionally span schema boundaries. A piece "\t\n"
    // increments the tab count to fill the last field separator AND
    // immediately ends the line. The mask's intra-piece walk inside
    // pieceFitsBudget allows this when the running tab count is exactly
    // tabsPerLineMax - 1 (so this tab fills the schema).
    // Sequence so far: "person_name\tJane\t0.92" — 2 tabs in.
    const pieces = ["person_name", "\t", "Jane", "0.92", "\t\n", "1"];
    const result = runMask({ pieces, generatedIds: [0, 1, 2, 1, 3] });
    assert.equal(result["\t\n"], "allowed", "composite tab+newline allowed when the tab fills the schema and the newline ends the line");
  });

  it("blocks a multi-character piece that would over-fill the tab budget", () => {
    // Sequence so far: "person_name\tJane\t0.92\t1" — 3 tabs. A
    // composite "\t\n" would push to 4 tabs before the newline reset,
    // which violates the schema mid-piece.
    const pieces = ["person_name", "\t", "Jane", "0.92", "1", "\t\n"];
    const result = runMask({ pieces, generatedIds: [0, 1, 2, 1, 3, 1, 4] });
    assert.equal(result["\t\n"], "blocked", "composite tab+newline blocked when tab would push past per-line budget");
  });

  it("returns a no-op when no tokenizer is provided", () => {
    const mask = createTsvGrammarMask({ fieldsPerLine: 4 });
    const logits = new Float32Array([0, 0, 0]);
    mask(logits, { generatedIds: [], vocabSize: 3 });
    // No tokenizer → mask should be a no-op (all zeros preserved).
    assert.equal(logits[0], 0);
    assert.equal(logits[1], 0);
    assert.equal(logits[2], 0);
  });
});
