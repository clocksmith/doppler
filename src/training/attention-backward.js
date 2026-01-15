import { acquireBuffer, uploadData, readBuffer } from '../gpu/buffer-pool.js';
import { createTensor, tensorBytes } from '../gpu/tensor.js';
import { f16ToF32Array } from '../inference/kv-cache/types.js';

function toFloat32(buffer, dtype) {
  if (dtype === 'f16') {
    return f16ToF32Array(new Uint16Array(buffer));
  }
  return new Float32Array(buffer);
}

function computeSoftmax(qData, kData, options) {
  const { seqLen, numHeads, headDim, scale = 1.0, causal = false } = options;
  const sData = new Float32Array(numHeads * seqLen * seqLen);

  for (let h = 0; h < numHeads; h += 1) {
    const qOffset = h * seqLen * headDim;
    const kOffset = h * seqLen * headDim;
    const sOffset = h * seqLen * seqLen;

    for (let i = 0; i < seqLen; i += 1) {
      let rowMax = -Infinity;
      for (let j = 0; j < seqLen; j += 1) {
        if (causal && j > i) {
          continue;
        }
        let sum = 0.0;
        for (let d = 0; d < headDim; d += 1) {
          sum += qData[qOffset + i * headDim + d] * kData[kOffset + j * headDim + d];
        }
        const scaled = sum * scale;
        if (scaled > rowMax) {
          rowMax = scaled;
        }
      }

      let rowSum = 0.0;
      for (let j = 0; j < seqLen; j += 1) {
        if (causal && j > i) {
          continue;
        }
        let sum = 0.0;
        for (let d = 0; d < headDim; d += 1) {
          sum += qData[qOffset + i * headDim + d] * kData[kOffset + j * headDim + d];
        }
        const expVal = Math.exp(sum * scale - rowMax);
        sData[sOffset + i * seqLen + j] = expVal;
        rowSum += expVal;
      }

      const invSum = rowSum > 0 ? 1 / rowSum : 0;
      for (let j = 0; j < seqLen; j += 1) {
        if (causal && j > i) {
          continue;
        }
        sData[sOffset + i * seqLen + j] *= invSum;
      }
    }
  }

  return sData;
}

export async function buildAttentionSoftmaxCache(q, k, options) {
  const [qBuf, kBuf] = await Promise.all([readBuffer(q.buffer), readBuffer(k.buffer)]);
  const qData = toFloat32(qBuf, q.dtype);
  const kData = toFloat32(kBuf, k.dtype);
  const sData = computeSoftmax(qData, kData, options);
  const { seqLen, numHeads } = options;
  const outBuf = acquireBuffer(tensorBytes([numHeads, seqLen, seqLen], 'f32'), undefined, 'attn_softmax_cache');
  uploadData(outBuf, sData);
  return createTensor(outBuf, 'f32', [numHeads, seqLen, seqLen], 'attn_softmax_cache');
}

export async function attentionBackwardCpu(
  q,
  k,
  v,
  softmax,
  gradOutput,
  options
) {
  const { seqLen, numHeads, headDim, scale = 1.0, causal = false } = options;

  const buffers = [
    readBuffer(q.buffer),
    readBuffer(k.buffer),
    readBuffer(v.buffer),
    readBuffer(gradOutput.buffer),
  ];
  if (softmax) {
    buffers.splice(3, 0, readBuffer(softmax.buffer));
  }

  const results = await Promise.all(buffers);
  const qBuf = results[0];
  const kBuf = results[1];
  const vBuf = results[2];
  const sBuf = softmax ? results[3] : null;
  const dBuf = softmax ? results[4] : results[3];

  const qData = toFloat32(qBuf, q.dtype);
  const kData = toFloat32(kBuf, k.dtype);
  const vData = toFloat32(vBuf, v.dtype);
  const sData = softmax
    ? toFloat32(sBuf, softmax.dtype)
    : computeSoftmax(qData, kData, options);
  const dData = toFloat32(dBuf, gradOutput.dtype);

  const total = seqLen * numHeads * headDim;
  const dQ = new Float32Array(total);
  const dK = new Float32Array(total);
  const dV = new Float32Array(total);

  for (let h = 0; h < numHeads; h += 1) {
    const qOffset = h * seqLen * headDim;
    const kOffset = h * seqLen * headDim;
    const vOffset = h * seqLen * headDim;
    const sOffset = h * seqLen * seqLen;
    const dOffset = h * seqLen * headDim;

    // dV = S^T @ dO
    for (let j = 0; j < seqLen; j += 1) {
      for (let d = 0; d < headDim; d += 1) {
        let sum = 0.0;
        for (let i = 0; i < seqLen; i += 1) {
          if (causal && j > i) {
            continue;
          }
          const s = sData[sOffset + i * seqLen + j];
          const dO = dData[dOffset + i * headDim + d];
          sum += s * dO;
        }
        dV[vOffset + j * headDim + d] = sum;
      }
    }

    // dS = dO @ V^T
    const dS = new Float32Array(seqLen * seqLen);
    for (let i = 0; i < seqLen; i += 1) {
      for (let j = 0; j < seqLen; j += 1) {
        if (causal && j > i) {
          continue;
        }
        let sum = 0.0;
        for (let d = 0; d < headDim; d += 1) {
          const dO = dData[dOffset + i * headDim + d];
          const vVal = vData[vOffset + j * headDim + d];
          sum += dO * vVal;
        }
        dS[i * seqLen + j] = sum;
      }
    }

    // dQK = softmax backward
    const dQK = new Float32Array(seqLen * seqLen);
    for (let i = 0; i < seqLen; i += 1) {
      let rowSum = 0.0;
      for (let j = 0; j < seqLen; j += 1) {
        if (causal && j > i) {
          continue;
        }
        const s = sData[sOffset + i * seqLen + j];
        rowSum += s * dS[i * seqLen + j];
      }
      for (let j = 0; j < seqLen; j += 1) {
        if (causal && j > i) {
          continue;
        }
        const s = sData[sOffset + i * seqLen + j];
        dQK[i * seqLen + j] = s * (dS[i * seqLen + j] - rowSum);
      }
    }

    // dQ = dQK @ K
    for (let i = 0; i < seqLen; i += 1) {
      for (let d = 0; d < headDim; d += 1) {
        let sum = 0.0;
        for (let j = 0; j < seqLen; j += 1) {
          if (causal && j > i) {
            continue;
          }
          sum += dQK[i * seqLen + j] * kData[kOffset + j * headDim + d];
        }
        dQ[qOffset + i * headDim + d] = sum * scale;
      }
    }

    // dK = dQK^T @ Q
    for (let j = 0; j < seqLen; j += 1) {
      for (let d = 0; d < headDim; d += 1) {
        let sum = 0.0;
        for (let i = 0; i < seqLen; i += 1) {
          if (causal && j > i) {
            continue;
          }
          sum += dQK[i * seqLen + j] * qData[qOffset + i * headDim + d];
        }
        dK[kOffset + j * headDim + d] = sum * scale;
      }
    }
  }

  const qBufOut = acquireBuffer(tensorBytes([seqLen, numHeads, headDim], 'f32'), undefined, 'attn_backward_q');
  const kBufOut = acquireBuffer(tensorBytes([seqLen, numHeads, headDim], 'f32'), undefined, 'attn_backward_k');
  const vBufOut = acquireBuffer(tensorBytes([seqLen, numHeads, headDim], 'f32'), undefined, 'attn_backward_v');

  uploadData(qBufOut, dQ);
  uploadData(kBufOut, dK);
  uploadData(vBufOut, dV);

  return {
    gradQ: createTensor(qBufOut, 'f32', [seqLen, numHeads, headDim], 'attn_grad_q'),
    gradK: createTensor(kBufOut, 'f32', [seqLen, numHeads, headDim], 'attn_grad_k'),
    gradV: createTensor(vBufOut, 'f32', [seqLen, numHeads, headDim], 'attn_grad_v'),
  };
}
