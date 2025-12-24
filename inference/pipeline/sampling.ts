/**
 * Token sampling from logits with temperature, top-k, top-p, and repetition penalty.
 */

export interface SamplingOptions {
  temperature: number;
  topP: number;
  topK: number;
  decode?: (tokens: number[]) => string;
  debug?: boolean;
}

export interface TokenCandidate {
  token: number;
  prob: number;
}

export interface TopKResult {
  token: number;
  logit: number;
  prob: number;
  text: string;
}

export interface LogitStats {
  min: number;
  max: number;
  nanCount: number;
  infCount: number;
  top5: TopKResult[];
}

export function applyRepetitionPenalty(
  logits: Float32Array,
  previousTokens: number[],
  penalty: number
): void {
  if (penalty === 1.0) return;

  const seen = new Set(previousTokens.slice(-100));
  for (const token of seen) {
    if (token < logits.length) {
      logits[token] = logits[token] > 0
        ? logits[token] / penalty
        : logits[token] * penalty;
    }
  }
}

export function softmax(logits: Float32Array): Float32Array {
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

export function sample(logits: Float32Array, opts: SamplingOptions): number {
  const { temperature, topP, topK, decode, debug = false } = opts;

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
      console.log(`[Sampling] Greedy: id=${maxIdx} "${text}" logit=${maxVal.toFixed(4)}`);
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
  let candidates: TokenCandidate[] = [];
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
    const filtered: TokenCandidate[] = [];
    for (const c of candidates) {
      filtered.push(c);
      cumProb += c.prob;
      if (cumProb >= topP) break;
    }
    candidates = filtered;
  }

  // Renormalize
  const probSum = candidates.reduce((s, c) => s + c.prob, 0);
  for (const c of candidates) {
    c.prob /= probSum;
  }

  if (debug) {
    const top5 = candidates.slice(0, 5).map(c => {
      const text = decode?.([c.token]) ?? '?';
      return `"${text}"(${(c.prob * 100).toFixed(1)}%)`;
    });
    console.log(`[Sampling] Top-5 (temp=${temperature}, topK=${topK}, topP=${topP}): ${top5.join(', ')}`);
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

export function getTopK(
  logits: Float32Array,
  k = 5,
  decode?: (tokens: number[]) => string
): TopKResult[] {
  const probs = softmax(new Float32Array(logits));

  const indexed: { token: number; logit: number; prob: number }[] = [];
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

export function logitsSanity(
  logits: Float32Array,
  label: string,
  decode?: (tokens: number[]) => string
): LogitStats {
  let min = Infinity;
  let max = -Infinity;
  let nanCount = 0;
  let infCount = 0;

  // Debug: Compare logits for specific tokens
  // "▁blue" = 3730, "▁BLUENRG" = 77590, "▁sky" = 7217
  // Plus the garbage tokens to understand why they have high logits
  const debugTokens = [
    { id: 3730, name: 'blue' },
    { id: 77590, name: 'BLUENRG' },
    { id: 7217, name: 'sky' },
    { id: 9595, name: 'Blue' },
    { id: 51481, name: 'BLUE' },
    { id: 44821, name: 'Kaw' },  // Garbage output token
    { id: 84327, name: 'Мини' },  // Russian "Mini" - another garbage
    { id: 0, name: 'PAD' },  // Check padding token
    { id: 1, name: 'BOS' },  // Check BOS token
    { id: 2, name: 'EOS' },  // Check EOS token
  ];
  const debugLogits = debugTokens
    .filter(t => t.id < logits.length)
    .map(t => `${t.name}:${logits[t.id]?.toFixed(2)}`)
    .join(', ');
  console.log(`[Pipeline] ${label} specific: ${debugLogits}`);

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
  const top5Str = top5.map(t => `"${t.text}"(${(t.prob * 100).toFixed(1)}%)`).join(', ');

  console.log(`[Pipeline] ${label} logits: min=${min.toFixed(2)}, max=${max.toFixed(2)} | top-5: ${top5Str}`);

  if (nanCount > 0 || infCount > 0) {
    console.warn(`[Pipeline] ${label} logits have ${nanCount} NaN, ${infCount} Inf values`);
  }

  return { min, max, nanCount, infCount, top5 };
}
