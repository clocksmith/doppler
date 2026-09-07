// Incremental JSON object syntax constraint. Schema validation remains a
// separate caller-owned boundary; this mask does not enforce application keys.
// Tokenizers must decode individual pieces without trimming whitespace.

function initialState() {
  return { stack: [{ kind: 'root', expect: 'object' }], scalar: null };
}

function completeValue(state) {
  const parent = state.stack[state.stack.length - 1];
  parent.expect = parent.kind === 'root' ? 'done' : 'comma-or-end';
}

function beginValue(state, char) {
  if (char === '{' || char === '[') {
    state.stack.push({ kind: char === '{' ? 'object' : 'array', expect: char === '{' ? 'key-or-end' : 'value-or-end' });
  } else if (char === '"') state.scalar = { kind: 'string', role: 'value', escaped: false, unicode: 0 };
  else if (char === 't' || char === 'f' || char === 'n') state.scalar = { kind: 'literal', remaining: { t: 'rue', f: 'alse', n: 'ull' }[char] };
  else if (char === '-' || /[0-9]/.test(char)) state.scalar = { kind: 'number', phase: char === '-' ? 'sign' : char === '0' ? 'zero' : 'integer' };
  else return false;
  return true;
}

function advanceNumber(number, char) {
  const digit = /[0-9]/.test(char);
  if (number.phase === 'sign') {
    if (!digit) return 'invalid';
    number.phase = char === '0' ? 'zero' : 'integer';
  } else if (number.phase === 'dot') {
    if (!digit) return 'invalid';
    number.phase = 'fraction';
  } else if (number.phase === 'exponent' || number.phase === 'exponent-sign') {
    if (number.phase === 'exponent' && (char === '+' || char === '-')) number.phase = 'exponent-sign';
    else if (digit) number.phase = 'exponent-digits';
    else return 'invalid';
  } else if (digit) {
    if (number.phase === 'zero') return 'invalid';
  } else if (char === '.' && ['zero', 'integer'].includes(number.phase)) number.phase = 'dot';
  else if ((char === 'e' || char === 'E') && ['zero', 'integer', 'fraction'].includes(number.phase)) number.phase = 'exponent';
  else return /[\x20\t\r\n,\]}]/.test(char) ? 'complete' : 'invalid';
  return 'continue';
}

function advance(state, text) {
  for (const char of text) {
    const scalar = state.scalar;
    if (scalar?.kind === 'string') {
      if (scalar.unicode > 0) {
        if (!/[a-fA-F0-9]/.test(char)) return false;
        scalar.unicode -= 1;
      } else if (scalar.escaped) {
        scalar.escaped = false;
        if (char === 'u') scalar.unicode = 4;
        else if (!'"\\/bfnrt'.includes(char)) return false;
      } else if (char === '\\') scalar.escaped = true;
      else if (char === '"') {
        state.scalar = null;
        if (scalar.role === 'key') state.stack[state.stack.length - 1].expect = 'colon';
        else completeValue(state);
      } else if (char.codePointAt(0) < 32) return false;
      continue;
    }
    if (scalar?.kind === 'literal') {
      if (char !== scalar.remaining[0]) return false;
      scalar.remaining = scalar.remaining.slice(1);
      if (!scalar.remaining) { state.scalar = null; completeValue(state); }
      continue;
    }
    if (scalar?.kind === 'number') {
      const result = advanceNumber(scalar, char);
      if (result === 'invalid') return false;
      if (result === 'continue') continue;
      state.scalar = null;
      completeValue(state);
    }
    if (/[\x20\t\r\n]/.test(char)) continue;
    const current = state.stack[state.stack.length - 1];
    if (current.expect === 'done') return false;
    if (current.expect === 'object') {
      if (char !== '{') return false;
      if (!beginValue(state, char)) return false;
    } else if (current.expect === 'colon') {
      if (char !== ':') return false;
      current.expect = 'value';
    } else if (current.expect === 'key' || current.expect === 'key-or-end') {
      if (char === '}' && current.expect === 'key-or-end') { state.stack.pop(); completeValue(state); }
      else if (char === '"') state.scalar = { kind: 'string', role: 'key', escaped: false, unicode: 0 };
      else return false;
    } else if (current.expect === 'comma-or-end') {
      if (char === ',') current.expect = current.kind === 'object' ? 'key' : 'value';
      else if (char === (current.kind === 'object' ? '}' : ']')) { state.stack.pop(); completeValue(state); }
      else return false;
    } else if (current.expect === 'value-or-end' && char === ']') {
      state.stack.pop(); completeValue(state);
    } else if (!beginValue(state, char)) return false;
  }
  return true;
}

export function createJsonGrammarMask(opts = {}) {
  const pieceCache = new Map();
  const cacheBudget = Math.max(1024, Math.floor(Number(opts.cacheBudget) || 32768));
  let cachedTokenizer = null;

  function pieceAt(tokenizer, id) {
    if (pieceCache.has(id)) return pieceCache.get(id);
    if (pieceCache.size >= cacheBudget) pieceCache.clear();
    const piece = tokenizer.decode([id], true, false);
    if (typeof piece !== 'string') throw new Error('JSON constraint requires synchronous string token decoding');
    pieceCache.set(id, piece);
    return piece;
  }

  return function logitMask(logits, context) {
    const tokenizer = opts.tokenizer ?? context?.tokenizer;
    if (!tokenizer || typeof tokenizer.decode !== 'function') throw new Error('JSON constraint requires a tokenizer');
    if (!Array.isArray(context?.generatedIds)) throw new Error('JSON constraint requires generated token IDs');
    if (cachedTokenizer !== tokenizer) { pieceCache.clear(); cachedTokenizer = tokenizer; }
    const state = initialState();
    const prefix = tokenizer.decode(context.generatedIds, true, false);
    if (typeof prefix !== 'string' || !advance(state, prefix)) throw new Error('JSON constraint received an invalid generated prefix');
    const complete = state.stack.length === 1 && state.stack[0].expect === 'done';
    const eos = tokenizer.getSpecialTokens?.().eos;
    const stopIds = new Set(opts.stopTokenIds ?? (Array.isArray(eos) ? eos : Number.isInteger(eos) ? [eos] : []));
    if (![...stopIds].every(id => Number.isInteger(id) && id >= 0 && id < logits.length)) throw new Error('JSON constraint has invalid stop token IDs');
    for (let id = 0; id < logits.length; id += 1) {
      if (logits[id] === -Infinity) continue;
      if (stopIds.has(id)) { if (!complete) logits[id] = -Infinity; continue; }
      const piece = pieceAt(tokenizer, id);
      const candidate = { stack: state.stack.map(row => ({ ...row })), scalar: state.scalar ? { ...state.scalar } : null };
      if (!piece || !advance(candidate, piece)) logits[id] = -Infinity;
    }
  };
}
