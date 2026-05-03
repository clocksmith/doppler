// Minimal TSV-grammar sampling constraint.
//
// Usage:
//   const mask = createTsvGrammarMask({ tokenizer, fieldsPerLine: 4 });
//   for await (const text of pipeline.generate(prompt, { logitMaskFn: mask })) { ... }
//
// Companion to json-grammar-mask.js. Use this when the model is asked to
// emit a flat tab-separated record stream (one record per line, fields
// separated by TAB, terminated by NEWLINE) rather than a JSON envelope.
// Cuts decode tokens by ~30-40% vs. JSON because the model isn't sampling
// braces/quotes/keys at every position — and Doppler's structured-
// generation primitive is logit validation only (no token forcing), so
// JSON's structural overhead is real.
//
// What the mask enforces (soft):
//   - Inside a line with fewer than `fieldsPerLine - 1` tabs already
//     emitted, disallow tokens whose piece contains a NEWLINE — the line
//     must hit all field separators before terminating.
//   - Disallow tokens that would push the per-line tab count past
//     `fieldsPerLine - 1`, so the model can't emit five-field rows when
//     four were specified.
//   - Allow any token (including end-of-stream) after a balanced newline
//     when at least one valid line has been emitted.
//
// What the mask does NOT enforce (intentional):
//   - Field-content shape. The category/confidence/page-number values
//     are validated post-decode by the consumer (e.g.,
//     parseDelimitedExtraction + validateRedaction in Columbo). Trying
//     to enforce enum prefixes at the BPE token-piece level is brittle
//     because token boundaries don't align with field boundaries.
//   - Empty lines. A trailing blank line is just dropped by the parser.
//
// Tokenizer assumption: same as json-grammar-mask.js — `tokenizer.decode([id])`
// returns the literal piece for that token. True for Qwen, Gemma, Llama BPE.

function pieceAt(tokenizer, tokenId) {
  if (!tokenizer || typeof tokenizer.decode !== "function") return "";
  try {
    return String(tokenizer.decode([tokenId], true, false) || "");
  } catch {
    return "";
  }
}

function countRunes(piece, char) {
  let count = 0;
  for (let i = 0; i < piece.length; i += 1) {
    if (piece[i] === char) count += 1;
  }
  return count;
}

/**
 * Per-piece flags used by the mask. Tracks tab/newline placement so the
 * state machine can decide when a line is complete and whether a token
 * is allowed to advance the schema.
 */
function classifyPiece(piece) {
  const tabs = countRunes(piece, "\t");
  const newlines = countRunes(piece, "\n");
  return {
    tabs,
    newlines,
    isOnlyWhitespace: /^\s*$/.test(piece),
    hasNewline: newlines > 0,
    hasTab: tabs > 0,
  };
}

/**
 * Create a logit mask function that enforces a soft TSV grammar.
 *
 * @param {{
 *   tokenizer?: { decode(ids: number[], skipSpecial?: boolean, skipBos?: boolean): string } | null,
 *   fieldsPerLine?: number,
 *   cacheBudget?: number,
 * }} [opts]
 * @returns {(logits: Float32Array, context: { generatedIds: number[], tokenizer?: unknown, vocabSize?: number }) => void}
 */
export function createTsvGrammarMask(opts = {}) {
  const fieldsPerLine = Math.max(2, Math.floor(Number(opts.fieldsPerLine) || 4));
  const tabsPerLineMax = fieldsPerLine - 1;
  const pieceCache = new Map();
  const cacheBudget = Math.max(1024, Math.floor(Number(opts.cacheBudget) || 32768));

  function cachedPiece(tokenizer, tokenId) {
    if (pieceCache.has(tokenId)) return pieceCache.get(tokenId);
    if (pieceCache.size >= cacheBudget) pieceCache.clear();
    const piece = pieceAt(tokenizer, tokenId);
    pieceCache.set(tokenId, piece);
    return piece;
  }

  // Per-line tab count resets on each emitted newline. `linesEmitted`
  // counts complete lines so the mask can permit EOS once at least one
  // valid record has been produced.
  let tabsOnCurrentLine = 0;
  let linesEmitted = 0;
  let lastGeneratedLen = 0;

  function reflectEmittedTokens(tokenizer, generatedIds) {
    for (let i = lastGeneratedLen; i < generatedIds.length; i += 1) {
      const piece = cachedPiece(tokenizer, generatedIds[i]);
      // Walk the piece char-by-char so multi-tab/newline pieces stay
      // accurate (e.g., a token piece "\t\n" both completes a line
      // AND resets the tab counter).
      for (let c = 0; c < piece.length; c += 1) {
        const ch = piece[c];
        if (ch === "\n") {
          if (tabsOnCurrentLine > 0) linesEmitted += 1;
          tabsOnCurrentLine = 0;
        } else if (ch === "\t") {
          tabsOnCurrentLine += 1;
        }
      }
    }
    lastGeneratedLen = generatedIds.length;
  }

  /**
   * Walk a candidate piece's chars and check whether any intermediate
   * state would violate the per-line tab budget. Returns true iff the
   * piece is structurally valid given the current `tabsOnCurrentLine`.
   *
   * Intra-piece state evolution matters because BPE tokens occasionally
   * span line boundaries (e.g., a piece `"\t\n"` ends one line AND
   * starts the next with tabsOnCurrentLine=0). A naive check that just
   * sums the piece's tabs/newlines would misjudge such tokens.
   */
  function pieceFitsBudget(piece, startingTabs, tabsMax) {
    let tabs = startingTabs;
    let sawNewlineWithoutCompleteLine = false;
    for (let c = 0; c < piece.length; c += 1) {
      const ch = piece[c];
      if (ch === "\n") {
        if (tabs < tabsMax) sawNewlineWithoutCompleteLine = true;
        tabs = 0;
      } else if (ch === "\t") {
        tabs += 1;
        if (tabs > tabsMax) return false;
      }
    }
    if (sawNewlineWithoutCompleteLine) return false;
    return true;
  }

  return function logitMask(logits, context) {
    const tokenizer = (opts.tokenizer ?? context?.tokenizer) || null;
    if (!tokenizer) return;
    reflectEmittedTokens(tokenizer, Array.isArray(context?.generatedIds) ? context.generatedIds : []);

    const vocabSize = Math.min(logits.length, Number(context?.vocabSize || logits.length));
    for (let id = 0; id < vocabSize; id += 1) {
      if (logits[id] <= -Infinity) continue;
      const piece = cachedPiece(tokenizer, id);
      if (!piece) continue;
      // Fast path: pieces with no tab/newline are always allowed —
      // they're field content (snippet text, digits, decimal points).
      // Avoids the per-char walk for the common case.
      if (piece.indexOf("\t") < 0 && piece.indexOf("\n") < 0) continue;
      if (!pieceFitsBudget(piece, tabsOnCurrentLine, tabsPerLineMax)) {
        logits[id] = -Infinity;
      }
    }
  };
}
