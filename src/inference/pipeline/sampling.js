/**
 * Token sampling from logits with temperature, top-k, top-p, and repetition penalty.
 */

import { log, trace, isTraceEnabled } from '../../debug/index.js';
import { getRuntimeConfig } from '../../config/runtime.js';

/**
 * @param {Float32Array} logits
 * @param {number[]} previousTokens
 * @param {number} penalty
 */
export function applyRepetitionPenalty(logits, previousTokens, penalty) {
  if (penalty === 1.0) return;

  const window = getRuntimeConfig().inference.sampling.repetitionPenaltyWindow;
  const seen = new Set(previousTokens.slice(-window));
  for (const token of seen) {
    if (token < logits.length) {
      logits[token] = logits[token] > 0
        ? logits[token] / penalty
        : logits[token] * penalty;
    }
  }
}

/**
 * @param {Float32Array} logits
 * @returns {Float32Array}
 */
export function softmax(logits) {
  const n = logits.length;
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    if (logits[i] > max) max = logits[i];
  }

  const exps = new Float32Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(logits[i] - max);
    exps[i] = e;
    sum += e;
  }

  const invSum = sum > 0 ? 1 / sum : 0;
  for (let i = 0; i < n; i++) {
    exps[i] *= invSum;
  }
  return exps;
}

/**
 * @param {Float32Array} logits
 * @param {import('./sampling.js').SamplingOptions} opts
 * @returns {number}
 */
export function sample(logits, opts) {
  const { temperature, topP, topK, decode, debug = false, padTokenId } = opts;

  if (padTokenId !== undefined && padTokenId >= 0 && padTokenId < logits.length) {
    logits[padTokenId] = -Infinity;
  }

  // Greedy (argmax) when temperature = 0
  if (temperature === 0) {
    let maxIdx = 0;
    let maxVal = logits[0];
    for (let i = 1; i < logits.length; i++) {
      if (logits[i] > maxVal) {
        maxVal = logits[i];
        maxIdx = i;
      }
    }
    if (debug) {
      const text = decode?.([maxIdx]) ?? '?';
      trace.sample(`Greedy: id=${maxIdx} "${text}" logit=${maxVal.toFixed(4)}`);
    }
    return maxIdx;
  }

  // Apply temperature
  if (temperature !== 1.0) {
    for (let i = 0; i < logits.length; i++) {
      logits[i] /= temperature;
    }
  }

  const probs = softmax(logits);

  // Build candidate list
  /** @type {import('./sampling.js').TokenCandidate[]} */
  let candidates = [];
  for (let i = 0; i < probs.length; i++) {
    candidates.push({ token: i, prob: probs[i] });
  }
  candidates.sort((a, b) => b.prob - a.prob);

  // Top-k filtering
  if (topK > 0) {
    candidates = candidates.slice(0, topK);
  }

  // Top-p (nucleus) filtering
  if (topP < 1.0) {
    let cumProb = 0;
    /** @type {import('./sampling.js').TokenCandidate[]} */
    const filtered = [];
    for (const c of candidates) {
      filtered.push(c);
      cumProb += c.prob;
      if (cumProb >= topP) break;
    }
    candidates = filtered;
  }

  // Renormalize with guard against zero sum
  const probSum = candidates.reduce((s, c) => s + c.prob, 0);
  if (probSum > 0) {
    for (const c of candidates) {
      c.prob /= probSum;
    }
  } else {
    // If all probabilities are zero, fall back to uniform distribution
    const uniformProb = 1.0 / candidates.length;
    for (const c of candidates) {
      c.prob = uniformProb;
    }
  }

  if (debug) {
    const top5 = candidates.slice(0, 5).map(c => {
      const text = decode?.([c.token]) ?? '?';
      return `"${text}"(${(c.prob * 100).toFixed(1)}%)`;
    });
    trace.sample(`Top-5 (temp=${temperature}, topK=${topK}, topP=${topP}): ${top5.join(', ')}`);
  }

  // Sample from distribution
  const r = Math.random();
  let cumProb = 0;
  for (const c of candidates) {
    cumProb += c.prob;
    if (r < cumProb) return c.token;
  }

  return candidates[candidates.length - 1].token;
}

/**
 * @param {Float32Array} logits
 * @param {number} [k]
 * @param {((tokens: number[]) => string) | undefined} [decode]
 * @returns {import('./sampling.js').TopKResult[]}
 */
export function getTopK(logits, k = 5, decode) {
  const probs = softmax(new Float32Array(logits));

  /** @type {{ token: number; logit: number; prob: number }[]} */
  const indexed = [];
  for (let i = 0; i < logits.length; i++) {
    indexed.push({ token: i, logit: logits[i], prob: probs[i] });
  }
  indexed.sort((a, b) => b.logit - a.logit);

  return indexed.slice(0, k).map(t => ({
    token: t.token,
    logit: t.logit,
    prob: t.prob,
    text: decode?.([t.token]) ?? `[${t.token}]`,
  }));
}

/**
 * @param {Float32Array} logits
 * @param {string} label
 * @param {((tokens: number[]) => string) | undefined} [decode]
 * @returns {import('./sampling.js').LogitStats}
 */
export function logitsSanity(logits, label, decode) {
  let min = Infinity;
  let max = -Infinity;
  let nanCount = 0;
  let infCount = 0;

  for (let i = 0; i < logits.length; i++) {
    const v = logits[i];
    if (Number.isNaN(v)) {
      nanCount++;
    } else if (!Number.isFinite(v)) {
      infCount++;
    } else {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }

  const top5 = getTopK(logits, 5, decode);
  if (isTraceEnabled('sample')) {
    const top5Str = top5.map(t => `"${t.text}"(${(t.prob * 100).toFixed(1)}%)`).join(', ');
    trace.sample(`${label} logits: min=${min.toFixed(2)}, max=${max.toFixed(2)} | top-5: ${top5Str}`);
  }

  if (nanCount > 0 || infCount > 0) {
    log.warn('Sampling', `${label} logits have ${nanCount} NaN, ${infCount} Inf values`);
  }

  return { min, max, nanCount, infCount, top5 };
}
