import { createRng, sampleNormal } from './rng.js';

function gelu(x) {
  const x3 = x * x * x;
  const inner = 0.7978845608028654 * (x + 0.044715 * x3);
  return 0.5 * x * (1.0 + Math.tanh(inner));
}

function clampScore(value) {
  if (!Number.isFinite(value)) return 0.0;
  if (value > 1e6) return 1e6;
  if (value < -1e6) return -1e6;
  return value;
}

function xavierStd(fanIn, fanOut) {
  const denom = fanIn + fanOut;
  return denom > 0 ? Math.sqrt(2.0 / denom) : 0.0;
}

export function createMlp(inputSize, hiddenSize, seed) {
  const safeInput = Math.max(1, Math.floor(inputSize));
  const safeHidden = Math.max(1, Math.floor(hiddenSize));
  const rng = createRng(Number.isFinite(seed) ? seed : 1337);

  const w1 = new Float32Array(safeInput * safeHidden);
  const w2 = new Float32Array(safeHidden);

  const w1Std = xavierStd(safeInput, safeHidden);
  const w2Std = xavierStd(safeHidden, 1);

  for (let i = 0; i < w1.length; i++) {
    w1[i] = sampleNormal(rng) * w1Std;
  }
  for (let i = 0; i < w2.length; i++) {
    w2[i] = sampleNormal(rng) * w2Std;
  }

  return {
    inputSize: safeInput,
    hiddenSize: safeHidden,
    w1,
    w2,
  };
}

export function mlpParamCount(mlp) {
  if (!mlp) return 0;
  return (mlp.w1?.length ?? 0) + (mlp.w2?.length ?? 0);
}

export function mlpToFlat(mlp) {
  const total = mlpParamCount(mlp);
  const out = new Float32Array(total);
  let offset = 0;
  if (mlp?.w1) {
    out.set(mlp.w1, offset);
    offset += mlp.w1.length;
  }
  if (mlp?.w2) {
    out.set(mlp.w2, offset);
  }
  return out;
}

export function mlpFromFlat(flat, inputSize, hiddenSize) {
  const safeInput = Math.max(1, Math.floor(inputSize));
  const safeHidden = Math.max(1, Math.floor(hiddenSize));
  const w1Count = safeInput * safeHidden;
  const w2Count = safeHidden;
  const required = w1Count + w2Count;
  if (!flat || flat.length < required) {
    throw new Error(`mlpFromFlat: expected ${required} params, got ${flat?.length ?? 0}.`);
  }

  const w1 = new Float32Array(w1Count);
  const w2 = new Float32Array(w2Count);
  w1.set(flat.subarray(0, w1Count));
  w2.set(flat.subarray(w1Count, w1Count + w2Count));
  return {
    inputSize: safeInput,
    hiddenSize: safeHidden,
    w1,
    w2,
  };
}

export function mlpForward(mlp, features) {
  if (!mlp || !features) return 0.0;
  const inputSize = mlp.inputSize;
  const hiddenSize = mlp.hiddenSize;
  if (!Number.isFinite(inputSize) || !Number.isFinite(hiddenSize)) return 0.0;
  if (features.length < inputSize) return 0.0;

  let score = 0.0;
  for (let h = 0; h < hiddenSize; h++) {
    let sum = 0.0;
    for (let i = 0; i < inputSize; i++) {
      sum += features[i] * mlp.w1[i * hiddenSize + h];
    }
    const act = gelu(sum);
    score += act * mlp.w2[h];
  }
  return clampScore(score);
}

export function mlpForwardBatch(mlp, featureRows, numRows) {
  if (!mlp || !featureRows) return new Float32Array(0);
  const inputSize = mlp.inputSize;
  const hiddenSize = mlp.hiddenSize;
  const rows = Math.max(0, Math.floor(numRows ?? 0));
  const out = new Float32Array(rows);
  if (!rows || inputSize <= 0 || hiddenSize <= 0) return out;
  if (featureRows.length < rows * inputSize) return out;

  for (let row = 0; row < rows; row++) {
    const base = row * inputSize;
    let score = 0.0;
    for (let h = 0; h < hiddenSize; h++) {
      let sum = 0.0;
      for (let i = 0; i < inputSize; i++) {
        sum += featureRows[base + i] * mlp.w1[i * hiddenSize + h];
      }
      const act = gelu(sum);
      score += act * mlp.w2[h];
    }
    out[row] = clampScore(score);
  }
  return out;
}

export function perturbMlp(mlp, rng, count, scale) {
  if (!mlp) return null;
  const safeCount = Math.max(1, Math.floor(count ?? 1));
  const safeScale = Number.isFinite(scale) ? scale : 1.0;
  const rand = typeof rng === 'function' ? rng : createRng(1337);

  const flat = mlpToFlat(mlp);
  for (let i = 0; i < safeCount; i++) {
    const idx = Math.floor(rand() * flat.length);
    flat[idx] += sampleNormal(rand) * safeScale;
  }
  return mlpFromFlat(flat, mlp.inputSize, mlp.hiddenSize);
}

