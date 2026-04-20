

import { log, trace, isTraceEnabled } from '../../../debug/index.js';
import { getRuntimeConfig } from '../../../config/runtime.js';

function seededRandom(seed) {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

let fallbackRandomState = (Date.now() >>> 0) || 0x6d2b79f5;

function unseededRandom() {
  if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
    const values = new Uint32Array(1);
    crypto.getRandomValues(values);
    return values[0] / 4294967296;
  }
  fallbackRandomState = (fallbackRandomState + 0x6d2b79f5) >>> 0;
  return fallbackRandomState / 4294967296;
}


export function applyRepetitionPenalty(logits, previousTokens, penalty) {
  if (penalty === 1.0) return;

  const windowSize = getRuntimeConfig().inference.sampling.repetitionPenaltyWindow;
  const seen = new Set(previousTokens.slice(-windowSize));
  for (const token of seen) {
    if (token < logits.length) {
      logits[token] = logits[token] > 0
        ? logits[token] / penalty
        : logits[token] * penalty;
    }
  }
}


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

function countFiniteCandidates(logits, padTokenId) {
  let finiteCandidateCount = 0;
  for (let i = 0; i < logits.length; i++) {
    if (padTokenId != null && i === padTokenId) {
      continue;
    }
    if (Number.isFinite(logits[i])) {
      finiteCandidateCount += 1;
    }
  }
  return finiteCandidateCount;
}

function assertFiniteSamplingCandidates(logits, padTokenId, label) {
  const finiteCandidateCount = countFiniteCandidates(logits, padTokenId);
  if (finiteCandidateCount > 0) {
    return;
  }
  throw new Error(
    `[Sampling] ${label} has no finite candidate logits after masking the pad token. ` +
    'Upstream decode likely produced NaN/Inf or an all-masked distribution.'
  );
}

function isHigherPriorityCandidate(candidate, current) {
  return candidate.logit > current.logit
    || (candidate.logit === current.logit && candidate.token < current.token);
}

function isLowerPriorityCandidate(candidate, current) {
  return candidate.logit < current.logit
    || (candidate.logit === current.logit && candidate.token > current.token);
}

function heapSwap(heap, a, b) {
  const tmp = heap[a];
  heap[a] = heap[b];
  heap[b] = tmp;
}

function heapPushLowestFirst(heap, candidate) {
  heap.push(candidate);
  let index = heap.length - 1;
  while (index > 0) {
    const parent = Math.floor((index - 1) / 2);
    if (!isLowerPriorityCandidate(heap[index], heap[parent])) {
      break;
    }
    heapSwap(heap, index, parent);
    index = parent;
  }
}

function heapifyLowestFirst(heap, index) {
  while (true) {
    const left = index * 2 + 1;
    const right = left + 1;
    let smallest = index;
    if (left < heap.length && isLowerPriorityCandidate(heap[left], heap[smallest])) {
      smallest = left;
    }
    if (right < heap.length && isLowerPriorityCandidate(heap[right], heap[smallest])) {
      smallest = right;
    }
    if (smallest === index) {
      return;
    }
    heapSwap(heap, index, smallest);
    index = smallest;
  }
}

function selectTopKLogitCandidates(logits, topK) {
  const limit = Math.max(1, Math.floor(topK));
  if (!Number.isFinite(limit) || limit >= logits.length) {
    return null;
  }

  const heap = [];
  for (let token = 0; token < logits.length; token++) {
    const logit = logits[token];
    if (!Number.isFinite(logit)) {
      continue;
    }
    const candidate = { token, logit };
    if (heap.length < limit) {
      heapPushLowestFirst(heap, candidate);
      continue;
    }
    if (isHigherPriorityCandidate(candidate, heap[0])) {
      heap[0] = candidate;
      heapifyLowestFirst(heap, 0);
    }
  }
  if (heap.length === 0) {
    return [];
  }
  return heap.sort((a, b) => b.logit - a.logit || a.token - b.token);
}

function sampleFromLogitCandidates(candidates, temperature, seed, decode, debug, topK, topP) {
  if (candidates.length === 0) {
    throw new Error(
      '[Sampling] Top-k filtering produced no finite candidates. ' +
      'Upstream decode likely produced NaN/Inf logits.'
    );
  }
  if (candidates.length === 1) {
    return candidates[0].token;
  }

  const invTemperature = 1 / temperature;
  let maxScaled = -Infinity;
  for (const candidate of candidates) {
    const scaled = candidate.logit * invTemperature;
    candidate.scaled = scaled;
    if (scaled > maxScaled) {
      maxScaled = scaled;
    }
  }

  let sum = 0;
  for (const candidate of candidates) {
    const weight = Math.exp(candidate.scaled - maxScaled);
    candidate.prob = weight;
    sum += weight;
  }

  if (sum > 0) {
    const invSum = 1 / sum;
    for (const candidate of candidates) {
      candidate.prob *= invSum;
    }
  } else {
    const uniformProb = 1.0 / candidates.length;
    for (const candidate of candidates) {
      candidate.prob = uniformProb;
    }
  }

  if (debug) {
    const top5 = candidates.slice(0, 5).map(c => {
      const text = decode?.([c.token]) ?? '?';
      return `"${text}"(${(c.prob * 100).toFixed(1)}%)`;
    });
    trace.sample(`Top-5 (temp=${temperature}, topK=${topK}, topP=${topP}): ${top5.join(', ')}`);
  }

  const r = seed !== undefined ? seededRandom(seed) : unseededRandom();
  let cumProb = 0;
  for (const candidate of candidates) {
    cumProb += candidate.prob;
    if (r < cumProb) return candidate.token;
  }
  return candidates[candidates.length - 1].token;
}


export function sample(logits, opts) {
  const { temperature, topP, topK, decode, debug = false, padTokenId, seed } = opts;

  if (padTokenId !== undefined && padTokenId >= 0 && padTokenId < logits.length) {
    logits[padTokenId] = -Infinity;
  }

  assertFiniteSamplingCandidates(logits, padTokenId, 'Logits');

  // Greedy (argmax) when temperature = 0
  if (temperature === 0) {
    let maxIdx = -1;
    let maxVal = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      const value = logits[i];
      if (!Number.isFinite(value)) {
        continue;
      }
      if (value > maxVal) {
        maxVal = value;
        maxIdx = i;
      }
    }
    if (maxIdx < 0) {
      throw new Error(
        '[Sampling] Greedy sampling could not find a finite candidate logit. ' +
        'Upstream decode likely produced NaN/Inf.'
      );
    }
    if (debug) {
      const text = decode?.([maxIdx]) ?? '?';
      trace.sample(`Greedy: id=${maxIdx} "${text}" logit=${maxVal.toFixed(4)}`);
    }
    return maxIdx;
  }

  if (topP >= 1.0 && Number.isFinite(topK) && topK > 0) {
    const candidates = selectTopKLogitCandidates(logits, topK);
    if (candidates) {
      return sampleFromLogitCandidates(candidates, temperature, seed, decode, debug, topK, topP);
    }
  }

  // Apply temperature
  if (temperature !== 1.0) {
    for (let i = 0; i < logits.length; i++) {
      logits[i] /= temperature;
    }
  }

  const probs = softmax(logits);

  // Build candidate list

  let candidates = [];
  for (let i = 0; i < probs.length; i++) {
    const probability = probs[i];
    if (!Number.isFinite(probability) || probability <= 0) {
      continue;
    }
    candidates.push({ token: i, prob: probability });
  }
  if (candidates.length === 0) {
    throw new Error(
      '[Sampling] Softmax produced no finite candidate probabilities. ' +
      'Upstream decode likely produced NaN/Inf logits.'
    );
  }
  candidates.sort((a, b) => b.prob - a.prob);

  // Top-k filtering
  if (topK > 0) {
    candidates = candidates.slice(0, topK);
  }

  // Top-p (nucleus) filtering
  if (topP < 1.0) {
    let cumProb = 0;

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
  const r = seed !== undefined ? seededRandom(seed) : unseededRandom();
  let cumProb = 0;
  for (const c of candidates) {
    cumProb += c.prob;
    if (r < cumProb) return c.token;
  }

  return candidates[candidates.length - 1].token;
}


export function getTopK(logits, k = 5, decode) {
  const probs = softmax(new Float32Array(logits));


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
