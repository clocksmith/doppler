import { createRng, sampleNormal } from './rng.js';

export function initWeights(mode, seed, scale, defaults) {
  const rng = createRng(seed);
  const safeScale = Number.isFinite(scale) ? scale : 1.0;
  const weights = new Float32Array(defaults.length);
  if (mode === 'zeros') {
    return weights;
  }
  for (let i = 0; i < defaults.length; i++) {
    weights[i] = defaults[i];
    if (mode === 'baseline') continue;
    if (mode === 'uniform') {
      weights[i] += (rng() * 2 - 1) * safeScale;
    } else {
      weights[i] += sampleNormal(rng) * safeScale;
    }
  }
  return weights;
}

export function perturbWeights(base, rng, count, scale) {
  const next = new Float32Array(base);
  const steps = Math.max(1, count);
  const safeScale = Number.isFinite(scale) ? scale : 1.0;
  for (let i = 0; i < steps; i++) {
    const idx = Math.floor(rng() * next.length);
    next[idx] += sampleNormal(rng) * safeScale;
  }
  return next;
}
