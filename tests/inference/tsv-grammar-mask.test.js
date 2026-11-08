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

  it("refuses to run without a tokenizer instead of degrading to a no-op", () => {
    // The pipelines charter requires token constraints to fail closed: a
    // constraint that silently allows every token is not a constraint.
    const mask = createTsvGrammarMask({ fieldsPerLine: 4 });
    const logits = new Float32Array([0, 0, 0]);
    assert.throws(() => mask(logits, { generatedIds: [], vocabSize: 3 }), /requires a tokenizer/);
    assert.deepEqual(Array.from(logits), [0, 0, 0], "logits are untouched when the constraint refuses");
  });
});

describe("createTsvGrammarMask fail-closed contract", () => {
  function tokenizerWithEos(pieces, eos) {
    return { decode(ids) { return ids.map((id) => pieces[id] ?? "").join(""); }, getSpecialTokens() { return { eos }; } };
  }

  it("blocks end-of-stream until one complete record has been emitted", () => {
    const pieces = ["name", "\t", "x", "\n", "<eos>"];
    const eos = 4;
    const mask = createTsvGrammarMask({ tokenizer: tokenizerWithEos(pieces, eos), fieldsPerLine: 2 });
    const run = (generatedIds) => { const logits = new Float32Array(pieces.length); mask(logits, { generatedIds, vocabSize: pieces.length }); return logits[eos] === -Infinity ? "blocked" : "allowed"; };
    assert.equal(run([]), "blocked", "nothing emitted yet");
    assert.equal(run([0, 1, 2]), "blocked", "a line is open");
    assert.equal(run([0, 1, 2, 3]), "allowed", "one complete two-field record, line closed");
    assert.equal(run([0, 1, 2, 3, 0]), "blocked", "a new line has started");
  });

  it("honours explicit stop token ids and rejects invalid ones", () => {
    const pieces = ["a", "\t", "\n", "<stop>"];
    const tokenizer = { decode(ids) { return ids.map((id) => pieces[id] ?? "").join(""); } };
    const mask = createTsvGrammarMask({ tokenizer, fieldsPerLine: 2, stopTokenIds: [3] });
    const logits = new Float32Array(pieces.length);
    mask(logits, { generatedIds: [], vocabSize: pieces.length });
    assert.equal(logits[3], -Infinity);
    const bad = createTsvGrammarMask({ tokenizer, fieldsPerLine: 2, stopTokenIds: [99] });
    assert.throws(() => bad(new Float32Array(pieces.length), { generatedIds: [], vocabSize: pieces.length }), /invalid stop token id 99/);
  });

  it("fails closed instead of allowing tokens it cannot decode", () => {
    const broken = { decode() { throw new Error("decode failed"); } };
    const mask = createTsvGrammarMask({ tokenizer: broken, fieldsPerLine: 2 });
    assert.throws(() => mask(new Float32Array(3), { generatedIds: [], vocabSize: 3 }), /decode failed/);
    const nonString = { decode() { return 7; } };
    assert.throws(() => createTsvGrammarMask({ tokenizer: nonString, fieldsPerLine: 2 })(new Float32Array(3), { generatedIds: [], vocabSize: 3 }), /synchronous string token decoding/);
    assert.throws(() => createTsvGrammarMask({ fieldsPerLine: 2 })(new Float32Array(3), { generatedIds: [] }), /requires a tokenizer/);
    assert.throws(() => createTsvGrammarMask({ tokenizer: broken, fieldsPerLine: 2 })(new Float32Array(3), {}), /requires generated token IDs/);
  });

  it("resets parse state when a shorter generation reuses the mask, and on demand", () => {
    const pieces = ["name", "\t", "x", "\n"];
    const tokenizer = { decode(ids) { return ids.map((id) => pieces[id] ?? "").join(""); } };
    const mask = createTsvGrammarMask({ tokenizer, fieldsPerLine: 2 });
    mask(new Float32Array(pieces.length), { generatedIds: [0, 1, 2, 3], vocabSize: pieces.length });
    assert.deepEqual(mask.snapshot(), { tabsOnCurrentLine: 0, linesEmitted: 1, reflectedTokens: 4 });
    // A new generation starts with a shorter prefix: stale state must not leak.
    const logits = new Float32Array(pieces.length);
    mask(logits, { generatedIds: [0], vocabSize: pieces.length });
    assert.deepEqual(mask.snapshot(), { tabsOnCurrentLine: 0, linesEmitted: 0, reflectedTokens: 1 });
    assert.equal(logits[3], -Infinity, "newline is blocked again at the start of a new record");
    mask.reset();
    assert.deepEqual(mask.snapshot(), { tabsOnCurrentLine: 0, linesEmitted: 0, reflectedTokens: 0 });
  });
});
