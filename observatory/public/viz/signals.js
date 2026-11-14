// Per-token signals for the answer strip, computed only from what a run recorded.
// Every value is null when its inputs were not recorded; nothing is estimated
// beyond what the label says.

export const DIVERSITY_WINDOW = 16;

// −ln p of the generated token. The probability is a full-vocabulary softmax, so this is exact.
export function surprise(token) {
  const p = token?.probability;
  return Number.isFinite(p) && p > 0 ? -Math.log(p) : null;
}

// Entropy of the recorded top candidates after renormalizing them to sum to one.
// It describes that subset only, not the model's full next-token distribution;
// `coverage` is how much of the full probability the subset held.
export function candidateSpread(token) {
  const listed = (token?.candidates ?? []).map((c) => c.probability).filter((p) => Number.isFinite(p) && p > 0);
  if (listed.length < 2) return null;
  const coverage = listed.reduce((sum, p) => sum + p, 0);
  const entropy = -listed.reduce((sum, p) => sum + (p / coverage) * Math.log(p / coverage), 0);
  return { entropy, max: Math.log(listed.length), count: listed.length, coverage };
}

// exp(mean surprise) over the answer tokens that recorded a probability, with the range used.
export function perplexity(tokens) {
  const values = tokens.map(surprise);
  const known = values.filter((value) => value !== null);
  if (!known.length) return null;
  const first = values.findIndex((value) => value !== null);
  const last = values.length - 1 - [...values].reverse().findIndex((value) => value !== null);
  return { value: Math.exp(known.reduce((sum, v) => sum + v, 0) / known.length), tokens: known.length, first, last };
}

// Share of distinct tokens among the last DIVERSITY_WINDOW answer tokens ending here,
// and whether this token already appeared in that window. A property of the text,
// not of the model's uncertainty.
export function diversity(tokens, index, window = DIVERSITY_WINDOW) {
  const start = Math.max(0, index - window + 1);
  const ids = tokens.slice(start, index + 1).map((token) => token.tokenId);
  return { distinct: new Set(ids).size / ids.length, repeated: ids.slice(0, -1).includes(ids.at(-1)), window: ids.length };
}

// Entropy of the full next-token distribution, in nats, when the runtime recorded it.
export function confusion(token) {
  return Number.isFinite(token?.entropy) && token.entropy >= 0 ? token.entropy : null;
}

export function answerSignals(tokens) {
  return tokens.map((token, index) => ({
    surprise: surprise(token),
    confusion: confusion(token),
    spread: candidateSpread(token),
    diversity: diversity(tokens, index),
  }));
}
