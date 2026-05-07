

import { dequantizeQ4_KBlockRef } from './dequant.js';

const Q4K_K = 256;
const Q4K_BLOCK_SIZE = 144;

function silu(x) {
  return x / (1 + Math.exp(-x));
}

function gelu(x) {
  const c = 0.7978845608;
  return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
}

function clampSwiglu(x, activationCode, clampMax) {
  if (clampMax <= 0 || activationCode !== 0) return x;
  if (x < -clampMax) return -clampMax;
  if (x > clampMax) return clampMax;
  return x;
}

export function dequantQ4KRowsRef(packed, numRows, K) {
  const numBlocksPerRow = Math.ceil(K / Q4K_K);
  const out = new Float32Array(numRows * K);
  for (let row = 0; row < numRows; row++) {
    for (let b = 0; b < numBlocksPerRow; b++) {
      const blockStart = (row * numBlocksPerRow + b) * Q4K_BLOCK_SIZE;
      const block = packed.subarray(blockStart, blockStart + Q4K_BLOCK_SIZE);
      const deq = dequantizeQ4_KBlockRef(block);
      const kBase = b * Q4K_K;
      const writeLen = Math.min(Q4K_K, K - kBase);
      for (let i = 0; i < writeLen; i++) {
        out[row * K + kBase + i] = deq[i];
      }
    }
  }
  return out;
}

export function fusedFfnQ4KRef({
  input,
  Wgate,
  Wup,
  M,
  hiddenSize,
  intermediateSize,
  alpha,
  activation,
  swigluLimit,
}) {
  const activationCode = activation === 'silu' ? 0 : 1;
  const clampMax = swigluLimit == null ? 0 : swigluLimit;
  const gateF32 = dequantQ4KRowsRef(Wgate, intermediateSize, hiddenSize);
  const upF32 = dequantQ4KRowsRef(Wup, intermediateSize, hiddenSize);
  const out = new Float32Array(M * intermediateSize);
  for (let m = 0; m < M; m++) {
    for (let col = 0; col < intermediateSize; col++) {
      let g = 0;
      let u = 0;
      const inBase = m * hiddenSize;
      const wBase = col * hiddenSize;
      for (let k = 0; k < hiddenSize; k++) {
        const a = input[inBase + k];
        g += a * gateF32[wBase + k];
        u += a * upF32[wBase + k];
      }
      const activated = activationCode === 0 ? silu(g) : gelu(g);
      out[m * intermediateSize + col] = clampSwiglu(activated * u * alpha, activationCode, clampMax);
    }
  }
  return out;
}
