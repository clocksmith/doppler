import { createRng, sampleNormal } from './rng.js';

export function initPriorities(count, mode, seed, scale) {
  const priorities = new Float32Array(count);
  const rng = createRng(seed);
  const safeScale = Number.isFinite(scale) ? scale : 1.0;
  if (mode === 'baseline') {
    return priorities;
  }
  if (mode === 'zeros') {
    return priorities;
  }
  for (let i = 0; i < count; i++) {
    if (mode === 'uniform') {
      priorities[i] = (rng() * 2 - 1) * safeScale;
    } else {
      priorities[i] = sampleNormal(rng) * safeScale;
    }
  }
  return priorities;
}

export function perturbPriorities(base, rng, count, scale) {
  const next = new Float32Array(base);
  const n = next.length;
  const steps = Math.max(1, count);
  for (let i = 0; i < steps; i++) {
    const idx = Math.floor(rng() * n);
    next[idx] += sampleNormal(rng) * scale;
  }
  return next;
}
