// TSV-grammar sampling constraint.
//
// Usage:
//   const mask = createTsvGrammarMask({ tokenizer, fieldsPerLine: 4 });
//   for await (const text of pipeline.generate(prompt, { logitMaskFn: mask })) { ... }
//
// Companion to json-grammar-mask.js. Use this when the model is asked to
// emit a flat tab-separated record stream (one record per line, fields
// separated by TAB, terminated by NEWLINE) rather than a JSON envelope.
// Cuts decode tokens by ~30-40% vs. JSON because the model isn't sampling
// braces/quotes/keys at every position, and Doppler's structured-
// generation primitive is logit validation only (no token forcing).
//
// What the mask enforces, failing closed (pipelines charter: token
// constraints fail closed and execute for every sampled token):
//   - Inside a line with fewer than `fieldsPerLine - 1` tabs already
//     emitted, disallow tokens whose piece contains a NEWLINE. The line
//     must hit all field separators before terminating.
//   - Disallow tokens that would push the per-line tab count past
//     `fieldsPerLine - 1`, so the model can't emit five-field rows when
//     four were specified.
//   - Disallow end-of-stream until at least one complete record has been
//     emitted and the current line is empty. Stop ids come from
//     `opts.stopTokenIds` or the tokenizer's EOS.
//   - A tokenizer that cannot decode a piece, or a missing tokenizer or
//     generated-id list, is an error: the constraint never degrades to
//     "allow everything".
//   - Parse state belongs to one generation. A shorter generated prefix
//     than the last one seen starts a new generation and resets the
//     state; `mask.reset()` does so explicitly.
//
// What the mask does NOT enforce (intentional):
//   - Field-content shape. The category/confidence/page-number values
//     are validated post-decode by the consumer. Enforcing enum prefixes
//     at the BPE token-piece level is brittle because token boundaries
//     don't align with field boundaries.
//   - Empty lines. A trailing blank line is just dropped by the parser.
//
// Tokenizer assumption: same as json-grammar-mask.js. `tokenizer.decode([id])`
// returns the literal piece for that token. True for Qwen, Gemma, Llama BPE.

function decodePiece(tokenizer, tokenId) {
  const piece = tokenizer.decode([tokenId], true, false);
  if (typeof piece !== 'string') {
    throw new Error(`TSV constraint requires synchronous string token decoding (token ${tokenId}).`);
  }
  return piece;
}

function resolveStopIds(opts, tokenizer, vocabSize) {
  const declared = opts.stopTokenIds;
  const eos = tokenizer.getSpecialTokens?.()?.eos;
  const candidates = Array.isArray(declared) ? declared
    : Array.isArray(eos) ? eos
      : Number.isInteger(eos) ? [eos] : [];
  for (const id of candidates) {
    if (!Number.isInteger(id) || id < 0 || id >= vocabSize) {
      throw new Error(`TSV constraint has an invalid stop token id ${String(id)} for vocabSize=${vocabSize}.`);
    }
  }
  return new Set(candidates);
}

export function createTsvGrammarMask(opts = {}) {
  const fieldsPerLine = Math.max(2, Math.floor(Number(opts.fieldsPerLine) || 4));
  const tabsPerLineMax = fieldsPerLine - 1;
  const pieceCache = new Map();
  const cacheBudget = Math.max(1024, Math.floor(Number(opts.cacheBudget) || 32768));

  function cachedPiece(tokenizer, tokenId) {
    if (pieceCache.has(tokenId)) return pieceCache.get(tokenId);
    if (pieceCache.size >= cacheBudget) pieceCache.clear();
    const piece = decodePiece(tokenizer, tokenId);
    pieceCache.set(tokenId, piece);
    return piece;
  }

  // Per-line tab count resets on each emitted newline. `linesEmitted`
  // counts complete lines so the mask can permit EOS once at least one
  // valid record has been produced.
  let tabsOnCurrentLine = 0;
  let lineHasContent = false;
  let linesEmitted = 0;
  let lastGeneratedLen = 0;

  function reset() {
    tabsOnCurrentLine = 0;
    lineHasContent = false;
    linesEmitted = 0;
    lastGeneratedLen = 0;
  }

  function reflectEmittedTokens(tokenizer, generatedIds) {
    // A shorter sequence than the one already reflected means a new
    // generation reused this mask: start the parse state over.
    if (generatedIds.length < lastGeneratedLen) reset();
    for (let i = lastGeneratedLen; i < generatedIds.length; i += 1) {
      const piece = cachedPiece(tokenizer, generatedIds[i]);
      // Walk the piece char-by-char so multi-tab/newline pieces stay
      // accurate (e.g., a token piece "\t\n" both completes a line
      // AND resets the tab counter).
      for (let c = 0; c < piece.length; c += 1) {
        const ch = piece[c];
        if (ch === '\n') {
          if (tabsOnCurrentLine === tabsPerLineMax) linesEmitted += 1;
          tabsOnCurrentLine = 0;
          lineHasContent = false;
        } else {
          lineHasContent = true;
          if (ch === '\t') tabsOnCurrentLine += 1;
        }
      }
    }
    lastGeneratedLen = generatedIds.length;
  }

  function pieceFitsBudget(piece, startingTabs, tabsMax) {
    let tabs = startingTabs;
    let sawNewlineWithoutCompleteLine = false;
    for (let c = 0; c < piece.length; c += 1) {
      const ch = piece[c];
      if (ch === '\n') {
        if (tabs < tabsMax) sawNewlineWithoutCompleteLine = true;
        tabs = 0;
      } else if (ch === '\t') {
        tabs += 1;
        if (tabs > tabsMax) return false;
      }
    }
    if (sawNewlineWithoutCompleteLine) return false;
    return true;
  }

  function logitMask(logits, context) {
    const tokenizer = (opts.tokenizer ?? context?.tokenizer) || null;
    if (!tokenizer || typeof tokenizer.decode !== 'function') {
      throw new Error('TSV constraint requires a tokenizer with synchronous decode.');
    }
    if (!Array.isArray(context?.generatedIds)) {
      throw new Error('TSV constraint requires generated token IDs.');
    }
    reflectEmittedTokens(tokenizer, context.generatedIds);

    const vocabSize = Math.min(logits.length, Number(context?.vocabSize || logits.length));
    const stopIds = resolveStopIds(opts, tokenizer, vocabSize);
    // End-of-stream is legal only between records: at least one complete
    // line emitted and nothing pending on the current line.
    const stopAllowed = linesEmitted >= 1 && !lineHasContent;
    for (let id = 0; id < vocabSize; id += 1) {
      if (logits[id] <= -Infinity) continue;
      if (stopIds.has(id)) {
        if (!stopAllowed) logits[id] = -Infinity;
        continue;
      }
      const piece = cachedPiece(tokenizer, id);
      if (!piece) continue;
      // Fast path: pieces with no tab/newline are always allowed. They are
      // field content (snippet text, digits, decimal points). Avoids the
      // per-char walk for the common case.
      if (piece.indexOf('\t') < 0 && piece.indexOf('\n') < 0) continue;
      if (!pieceFitsBudget(piece, tabsOnCurrentLine, tabsPerLineMax)) {
        logits[id] = -Infinity;
      }
    }
  }

  logitMask.reset = reset;
  logitMask.snapshot = () => ({ tabsOnCurrentLine, linesEmitted, reflectedTokens: lastGeneratedLen });
  return logitMask;
}
