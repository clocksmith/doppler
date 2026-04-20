import assert from 'node:assert/strict';

import { sample } from '../../src/inference/pipelines/text/sampling.js';

function seededRandom(seed) {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

function referenceTopKSample(logits, opts) {
  const { temperature, topK, seed } = opts;
  const candidates = [];
  for (let token = 0; token < logits.length; token++) {
    const logit = logits[token];
    if (Number.isFinite(logit)) {
      candidates.push({ token, logit });
    }
  }
  candidates.sort((a, b) => b.logit - a.logit || a.token - b.token);
  const filtered = candidates.slice(0, topK);
  const maxScaled = Math.max(...filtered.map((candidate) => candidate.logit / temperature));
  let sum = 0;
  for (const candidate of filtered) {
    candidate.prob = Math.exp((candidate.logit / temperature) - maxScaled);
    sum += candidate.prob;
  }
  for (const candidate of filtered) {
    candidate.prob /= sum;
  }
  const r = seededRandom(seed);
  let cumulative = 0;
  for (const candidate of filtered) {
    cumulative += candidate.prob;
    if (r < cumulative) return candidate.token;
  }
  return filtered.at(-1).token;
}

{
  const logits = new Float32Array([1.5, -0.2, 3.0, 2.8, 0.1, 2.6, -Infinity, 1.2]);
  const opts = { temperature: 0.8, topK: 3, topP: 1, seed: 42 };
  assert.equal(sample(new Float32Array(logits), opts), referenceTopKSample(logits, opts));
}

{
  const logits = new Float32Array([4, 3, 4, 2, 1]);
  assert.equal(
    sample(logits, { temperature: 1, topK: 1, topP: 1, seed: 3, padTokenId: 0 }),
    2
  );
}

console.log('sampling-topk-fast-path.test: ok');
