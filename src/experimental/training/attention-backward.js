import { acquireBuffer, uploadData, readBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor, tensorBytes } from '../../gpu/tensor.js';
import { f16ToF32Array } from '../../inference/kv-cache/types.js';
import { createUploadedTensor } from './tensor-factory.js';

function toFloat32(buffer, dtype) {
  if (dtype === 'f16') {
    return f16ToF32Array(new Uint16Array(buffer));
  }
  return new Float32Array(buffer);
}

function resolveAttentionGeometry(options) {
  const seqLen = Math.floor(Number(options.seqLen));
  const numHeads = Math.floor(Number(options.numHeads));
  const numKVHeads = Math.floor(Number(options.numKVHeads ?? numHeads));
  const headDim = Math.floor(Number(options.headDim));
  if (
    seqLen < 1
    || numHeads < 1
    || numKVHeads < 1
    || headDim < 1
    || numHeads % numKVHeads !== 0
  ) {
    throw new Error('attention backward requires valid GQA geometry.');
  }
  return {
    seqLen,
    numHeads,
    numKVHeads,
    headDim,
    headsPerKV: numHeads / numKVHeads,
    scale: Number.isFinite(options.scale) ? options.scale : 1,
    causal: options.causal === true,
  };
}

export function computeAttentionSoftmaxData(qData, kData, options) {
  const {
    seqLen,
    numHeads,
    numKVHeads,
    headDim,
    headsPerKV,
    scale,
    causal,
  } = resolveAttentionGeometry(options);
  const sData = new Float32Array(numHeads * seqLen * seqLen);

  for (let h = 0; h < numHeads; h += 1) {
    const kvHead = Math.floor(h / headsPerKV);
    const sOffset = h * seqLen * seqLen;

    for (let i = 0; i < seqLen; i += 1) {
      let rowMax = -Infinity;
      for (let j = 0; j < seqLen; j += 1) {
        if (causal && j > i) {
          continue;
        }
        let sum = 0.0;
        for (let d = 0; d < headDim; d += 1) {
          const qIndex = (i * numHeads + h) * headDim + d;
          const kIndex = (j * numKVHeads + kvHead) * headDim + d;
          sum += qData[qIndex] * kData[kIndex];
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
          const qIndex = (i * numHeads + h) * headDim + d;
          const kIndex = (j * numKVHeads + kvHead) * headDim + d;
          sum += qData[qIndex] * kData[kIndex];
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
  const sData = computeAttentionSoftmaxData(qData, kData, options);
  const { seqLen, numHeads } = options;
  return createUploadedTensor(sData, 'f32', [numHeads, seqLen, seqLen], 'attn_softmax_cache');
}

export async function attentionBackwardCpu(
  q,
  k,
  v,
  softmax,
  gradOutput,
  options
) {
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
    : computeAttentionSoftmaxData(qData, kData, options);
  const dData = toFloat32(dBuf, gradOutput.dtype);

  const { dQ, dK, dV, geometry } = computeAttentionBackwardData(
    qData,
    kData,
    vData,
    sData,
    dData,
    options
  );

  let qBufOut = null;
  let kBufOut = null;
  let vBufOut = null;
  try {
    qBufOut = acquireBuffer(tensorBytes(q.shape, 'f32'), undefined, 'attn_backward_q');
    kBufOut = acquireBuffer(tensorBytes(k.shape, 'f32'), undefined, 'attn_backward_k');
    vBufOut = acquireBuffer(tensorBytes(v.shape, 'f32'), undefined, 'attn_backward_v');

    uploadData(qBufOut, dQ);
    uploadData(kBufOut, dK);
    uploadData(vBufOut, dV);

    return {
      gradQ: createTensor(qBufOut, 'f32', [...q.shape], 'attn_grad_q'),
      gradK: createTensor(kBufOut, 'f32', [...k.shape], 'attn_grad_k'),
      gradV: createTensor(vBufOut, 'f32', [...v.shape], 'attn_grad_v'),
      geometry,
    };
  } catch (error) {
    if (qBufOut) {
      releaseBuffer(qBufOut);
    }
    if (kBufOut) {
      releaseBuffer(kBufOut);
    }
    if (vBufOut) {
      releaseBuffer(vBufOut);
    }
    throw error;
  }
}

export function computeAttentionBackwardData(
  qData,
  kData,
  vData,
  sData,
  dData,
  options
) {
  const geometry = resolveAttentionGeometry(options);
  const {
    seqLen,
    numHeads,
    numKVHeads,
    headDim,
    headsPerKV,
    scale,
    causal,
  } = geometry;

  const dQ = new Float32Array(seqLen * numHeads * headDim);
  const dK = new Float32Array(seqLen * numKVHeads * headDim);
  const dV = new Float32Array(seqLen * numKVHeads * headDim);

  for (let h = 0; h < numHeads; h += 1) {
    const kvHead = Math.floor(h / headsPerKV);
    const sOffset = h * seqLen * seqLen;

    // dV = S^T @ dO
    for (let j = 0; j < seqLen; j += 1) {
      for (let d = 0; d < headDim; d += 1) {
        let sum = 0.0;
        for (let i = 0; i < seqLen; i += 1) {
          if (causal && j > i) {
            continue;
          }
          const s = sData[sOffset + i * seqLen + j];
          const dO = dData[(i * numHeads + h) * headDim + d];
          sum += s * dO;
        }
        const vIndex = (j * numKVHeads + kvHead) * headDim + d;
        dV[vIndex] += sum;
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
          const dO = dData[(i * numHeads + h) * headDim + d];
          const vIndex = (j * numKVHeads + kvHead) * headDim + d;
          const vVal = vData[vIndex];
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
          const kIndex = (j * numKVHeads + kvHead) * headDim + d;
          sum += dQK[i * seqLen + j] * kData[kIndex];
        }
        const qIndex = (i * numHeads + h) * headDim + d;
        dQ[qIndex] = sum * scale;
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
          const qIndex = (i * numHeads + h) * headDim + d;
          sum += dQK[i * seqLen + j] * qData[qIndex];
        }
        const kIndex = (j * numKVHeads + kvHead) * headDim + d;
        dK[kIndex] += sum * scale;
      }
    }
  }

  return { dQ, dK, dV, geometry };
}
