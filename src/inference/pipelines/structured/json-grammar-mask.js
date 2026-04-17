// Minimal JSON-grammar sampling constraint.
//
// Usage:
//   const mask = createJsonGrammarMask({ tokenizer });
//   for await (const text of pipeline.generate(prompt, { logitMaskFn: mask })) { ... }
//
// The mask classifies each candidate token against the current parse state and
// zeroes logits for tokens that would break a valid JSON object output. It
// deliberately accepts any token that *looks like* it may extend valid JSON —
// this is a soft constraint; parsing the final output is still required to
// validate the full payload.
//
// Tokenizer assumption: `tokenizer.decode([id])` returns the literal string
// piece for that token. That's true for BPE tokenizers used by Qwen, Gemma,
// Llama.  The mask is deliberately token-piece based, not character level,
// because re-encoding the stream to characters per step is prohibitively slow
// in-browser.
//
// What the mask does enforce:
//   - Until the first "{" is emitted, only tokens whose piece contains "{" or
//     whitespace are allowed.
//   - Once inside the object, matched counts of "{" / "}" / "[" / "]" / "\""
//     determine whether we are inside a string; inside a string, any token is
//     allowed except one that closes more brackets than are open.
//   - After a balanced closing "}" is emitted at depth 0, we forbid any
//     further non-whitespace token (forces a clean stop).
//
// What the mask does NOT enforce (intentional, to keep the mask cheap):
//   - Key quoting, trailing-comma avoidance, matching " pairs at token level.
//     Post-decode `parseJSONObjectEnvelope` handles these via recovery
//     transformations.
//   - Schema/key whitelists. Callers can layer a stricter mask on top.

function pieceAt(tokenizer, tokenId) {
  if (!tokenizer || typeof tokenizer.decode !== "function") return "";
  try {
    // decode(ids, skipSpecial, skipBos)
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

// Piece-level flags used by the mask.
function classifyPiece(piece) {
  const openBraces = countRunes(piece, "{");
  const closeBraces = countRunes(piece, "}");
  const openBrackets = countRunes(piece, "[");
  const closeBrackets = countRunes(piece, "]");
  const quotes = countRunes(piece, "\"");
  return {
    openBraces,
    closeBraces,
    openBrackets,
    closeBrackets,
    quotes,
    isOnlyWhitespace: /^\s*$/.test(piece),
    hasOpenBrace: openBraces > 0,
  };
}

/**
 * Create a logit mask function that enforces a soft JSON-object grammar.
 *
 * @param {{ tokenizer?: { decode(ids: number[], skipSpecial?: boolean, skipBos?: boolean): string } | null, cacheBudget?: number }} [opts]
 * @returns {(logits: Float32Array, context: { generatedIds: number[], tokenizer?: unknown, vocabSize?: number }) => void}
 */
export function createJsonGrammarMask(opts = {}) {
  const pieceCache = new Map();
  const cacheBudget = Math.max(1024, Math.floor(Number(opts.cacheBudget) || 32768));

  function cachedPiece(tokenizer, tokenId) {
    if (pieceCache.has(tokenId)) return pieceCache.get(tokenId);
    if (pieceCache.size >= cacheBudget) pieceCache.clear();
    const piece = pieceAt(tokenizer, tokenId);
    pieceCache.set(tokenId, piece);
    return piece;
  }

  let runningOpenBraces = 0;
  let runningOpenBrackets = 0;
  let runningQuotes = 0;
  let sawFirstBrace = false;
  let balancedClose = false;
  let lastGeneratedLen = 0;

  function reflectEmittedTokens(tokenizer, generatedIds) {
    // Catch up to any tokens committed since we were last called.
    for (let i = lastGeneratedLen; i < generatedIds.length; i += 1) {
      const piece = cachedPiece(tokenizer, generatedIds[i]);
      const shape = classifyPiece(piece);
      runningOpenBraces += shape.openBraces - shape.closeBraces;
      runningOpenBrackets += shape.openBrackets - shape.closeBrackets;
      runningQuotes += shape.quotes;
      if (!sawFirstBrace && shape.hasOpenBrace) sawFirstBrace = true;
      if (sawFirstBrace && runningOpenBraces <= 0 && runningOpenBrackets <= 0) {
        balancedClose = true;
      }
    }
    lastGeneratedLen = generatedIds.length;
  }

  return function logitMask(logits, context) {
    const tokenizer = (opts.tokenizer ?? context?.tokenizer) || null;
    if (!tokenizer) return;
    reflectEmittedTokens(tokenizer, Array.isArray(context?.generatedIds) ? context.generatedIds : []);
    const insideString = runningQuotes % 2 === 1;
    const vocabSize = Math.min(logits.length, Number(context?.vocabSize || logits.length));
    for (let id = 0; id < vocabSize; id += 1) {
      if (logits[id] <= -Infinity) continue;
      const piece = cachedPiece(tokenizer, id);
      if (!piece) continue;
      const shape = classifyPiece(piece);
      let ok = true;
      if (!sawFirstBrace) {
        // Before the first "{", only allow whitespace or a piece that opens a brace.
        ok = shape.isOnlyWhitespace || shape.hasOpenBrace;
      } else if (balancedClose) {
        // After the balanced close, only whitespace is allowed (model should stop).
        ok = shape.isOnlyWhitespace;
      } else if (insideString) {
        // Inside a string literal, accept any piece — even ones with braces or
        // brackets — the parser handles them as content.
        ok = true;
      } else {
        // Outside strings: disallow pieces that close MORE brackets than are open.
        if (shape.closeBraces > runningOpenBraces + shape.openBraces) ok = false;
        if (shape.closeBrackets > runningOpenBrackets + shape.openBrackets) ok = false;
      }
      if (!ok) {
        logits[id] = -Infinity;
      }
    }
  };
}
