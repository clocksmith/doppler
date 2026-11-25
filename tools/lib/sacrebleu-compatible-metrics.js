// Corpus BLEU and chrF that reproduce sacrebleu 2.6.0 defaults in JavaScript.
//
// BLEU matches `sacrebleu.metrics.BLEU()`: tokenize 13a, smooth exp, mixed case,
// effective_order False, max n-gram order 4. chrF matches
// `sacrebleu.metrics.CHRF()`: char_order 6, word_order 0, beta 2, whitespace
// excluded, eps_smoothing False. Both take one reference per hypothesis.
// Python string semantics are kept where they change results: `str.split()` and
// `str.rstrip()` use Python's whitespace set, and character n-grams count code
// points rather than UTF-16 units.

export const SACREBLEU_COMPATIBLE_VERSION = '2.6.0';
export const SACREBLEU_COMPATIBLE_IMPLEMENTATION =
  `doppler-js-sacrebleu-${SACREBLEU_COMPATIBLE_VERSION}-compatible`;

export const BLEU_SIGNATURE =
  `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|impl:${SACREBLEU_COMPATIBLE_IMPLEMENTATION}`;
export const CHRF_SIGNATURE =
  `nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|impl:${SACREBLEU_COMPATIBLE_IMPLEMENTATION}`;

const BLEU_MAX_ORDER = 4;
const CHRF_CHAR_ORDER = 6;
const CHRF_BETA = 2;

// Characters for which Python's str.isspace() is true.
const PY_WHITESPACE_CLASS =
  '\\t\\n\\v\\f\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000';
const PY_WHITESPACE_RUN = new RegExp(`[${PY_WHITESPACE_CLASS}]+`, 'gu');
const PY_TRAILING_WHITESPACE = new RegExp(`[${PY_WHITESPACE_CLASS}]+$`, 'u');

// sacrebleu TokenizerRegexp rules, in order. The first class is
// [\{-\~\[-\` -\&\(-\+\:-\@\/] written with explicit code points.
const TOKENIZER_13A_RULES = [
  [/([\x7B-\x7E\x5B-\x60\x20-\x26\x28-\x2B\x3A-\x40\x2F])/gu, ' $1 '],
  [/([^0-9])([.,])/gu, '$1 $2 '],
  [/([.,])([^0-9])/gu, ' $1 $2'],
  [/([0-9])(-)/gu, '$1 $2 '],
];

function pySplit(text) {
  return text.split(PY_WHITESPACE_RUN).filter((part) => part.length > 0);
}

function pyRstrip(text) {
  return text.replace(PY_TRAILING_WHITESPACE, '');
}

function requireSegments(hypotheses, references, label) {
  if (!Array.isArray(hypotheses) || !Array.isArray(references)) {
    throw new Error(`${label}: hypotheses and references must be arrays.`);
  }
  if (hypotheses.length !== references.length) {
    throw new Error(
      `${label}: ${hypotheses.length} hypotheses but ${references.length} references.`
    );
  }
  for (let index = 0; index < hypotheses.length; index += 1) {
    if (typeof hypotheses[index] !== 'string') {
      throw new Error(`${label}: hypothesis ${index} must be a string.`);
    }
    if (typeof references[index] !== 'string') {
      throw new Error(`${label}: reference ${index} must be a string.`);
    }
  }
}

// sacrebleu Tokenizer13a.__call__ followed by TokenizerRegexp.__call__.
export function tokenize13a(line) {
  let text = line.replaceAll('<skipped>', '').replaceAll('-\n', '').replaceAll('\n', ' ');
  if (text.includes('&')) {
    text = text
      .replaceAll('&quot;', '"')
      .replaceAll('&amp;', '&')
      .replaceAll('&lt;', '<')
      .replaceAll('&gt;', '>');
  }
  text = ` ${text} `;
  for (const [pattern, replacement] of TOKENIZER_13A_RULES) {
    text = text.replace(pattern, replacement);
  }
  return pySplit(text).join(' ');
}

function countWordNgrams(tokens, maxOrder) {
  const counts = new Map();
  for (let order = 1; order <= maxOrder; order += 1) {
    for (let start = 0; start + order <= tokens.length; start += 1) {
      const key = `${order}\u0000${tokens.slice(start, start + order).join(' ')}`;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  }
  return counts;
}

function myLog(value) {
  return value === 0 ? -9999999999 : Math.log(value);
}

// sacrebleu BLEU.compute_bleu with smooth_method 'exp' and effective_order False.
function computeBleuFromStatistics(correct, total, sysLen, refLen) {
  const brevityPenalty = sysLen < refLen
    ? (sysLen > 0 ? Math.exp(1 - refLen / sysLen) : 0)
    : 1;
  const precisions = new Array(BLEU_MAX_ORDER).fill(0);
  if (!correct.some((value) => value !== 0)) {
    return { score: 0, precisions, brevityPenalty };
  }
  let smoothMteval = 1;
  for (let order = 1; order <= BLEU_MAX_ORDER; order += 1) {
    if (total[order - 1] === 0) {
      break;
    }
    if (correct[order - 1] === 0) {
      smoothMteval *= 2;
      precisions[order - 1] = 100 / (smoothMteval * total[order - 1]);
    } else {
      precisions[order - 1] = 100 * correct[order - 1] / total[order - 1];
    }
  }
  let logSum = 0;
  for (const precision of precisions) {
    logSum += myLog(precision);
  }
  return {
    score: brevityPenalty * Math.exp(logSum / BLEU_MAX_ORDER),
    precisions,
    brevityPenalty,
  };
}

export function corpusBleu(hypotheses, references) {
  requireSegments(hypotheses, references, 'corpusBleu');
  const correct = new Array(BLEU_MAX_ORDER).fill(0);
  const total = new Array(BLEU_MAX_ORDER).fill(0);
  let sysLen = 0;
  let refLen = 0;
  for (let index = 0; index < hypotheses.length; index += 1) {
    const hypTokens = pySplit(tokenize13a(pyRstrip(hypotheses[index])));
    const refTokens = pySplit(tokenize13a(pyRstrip(references[index])));
    const hypCounts = countWordNgrams(hypTokens, BLEU_MAX_ORDER);
    const refCounts = countWordNgrams(refTokens, BLEU_MAX_ORDER);
    sysLen += hypTokens.length;
    refLen += refTokens.length;
    for (const [key, count] of hypCounts) {
      const order = Number(key.slice(0, key.indexOf('\u0000')));
      total[order - 1] += count;
      const refCount = refCounts.get(key);
      if (refCount !== undefined) {
        correct[order - 1] += Math.min(count, refCount);
      }
    }
  }
  const result = computeBleuFromStatistics(correct, total, sysLen, refLen);
  return {
    score: result.score,
    signature: BLEU_SIGNATURE,
    counts: correct,
    totals: total,
    precisions: result.precisions,
    brevityPenalty: result.brevityPenalty,
    sysLen,
    refLen,
  };
}

function countCharNgrams(text) {
  const chars = Array.from(pySplit(text).join(''));
  const counters = [];
  for (let order = 1; order <= CHRF_CHAR_ORDER; order += 1) {
    const counts = new Map();
    for (let start = 0; start + order <= chars.length; start += 1) {
      const key = chars.slice(start, start + order).join('');
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    counters.push(counts);
  }
  return counters;
}

function sumCounts(counts) {
  let sum = 0;
  for (const value of counts.values()) {
    sum += value;
  }
  return sum;
}

// sacrebleu CHRF._compute_f_score with eps_smoothing False.
function computeChrfFromStatistics(statistics) {
  const factor = CHRF_BETA ** 2;
  let avgPrec = 0;
  let avgRec = 0;
  let effectiveOrder = 0;
  for (let order = 0; order < CHRF_CHAR_ORDER; order += 1) {
    const [nHyp, nRef, nMatch] = statistics.slice(3 * order, 3 * order + 3);
    if (nHyp > 0 && nRef > 0) {
      avgPrec += nMatch / nHyp;
      avgRec += nMatch / nRef;
      effectiveOrder += 1;
    }
  }
  if (effectiveOrder === 0) {
    avgPrec = 0;
    avgRec = 0;
  } else {
    avgPrec /= effectiveOrder;
    avgRec /= effectiveOrder;
  }
  if (avgPrec + avgRec === 0) {
    return 0;
  }
  let score = (1 + factor) * avgPrec * avgRec;
  score /= (factor * avgPrec) + avgRec;
  return 100 * score;
}

export function corpusChrf(hypotheses, references) {
  requireSegments(hypotheses, references, 'corpusChrf');
  const statistics = new Array(3 * CHRF_CHAR_ORDER).fill(0);
  for (let index = 0; index < hypotheses.length; index += 1) {
    const hypCounters = countCharNgrams(hypotheses[index]);
    const refCounters = countCharNgrams(references[index]);
    for (let order = 0; order < CHRF_CHAR_ORDER; order += 1) {
      const hypCounts = hypCounters[order];
      const refCounts = refCounters[order];
      let matches = 0;
      for (const [key, count] of hypCounts) {
        const refCount = refCounts.get(key);
        if (refCount !== undefined) {
          matches += Math.min(count, refCount);
        }
      }
      statistics[3 * order] += sumCounts(hypCounts);
      statistics[3 * order + 1] += sumCounts(refCounts);
      statistics[3 * order + 2] += matches;
    }
  }
  return {
    score: computeChrfFromStatistics(statistics),
    signature: CHRF_SIGNATURE,
    statistics,
  };
}
