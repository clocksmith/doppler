

import { releaseBuffer } from '../../../memory/buffer-pool.js';
import { isGpuBufferInstance, isWeightBuffer } from '../../../gpu/weight-buffer.js';
import { runMatmul, recordMatmul } from '../../../gpu/kernel-selector.js';
import { runResidualAdd, recordResidualAdd } from '../../../gpu/kernels/residual.js';
import { runScale, recordScale } from '../../../gpu/kernels/scale.js';

function getKnownElementCount(weight) {
  if (weight instanceof Float32Array || weight instanceof Uint16Array) {
    return weight.length;
  }
  if (isWeightBuffer(weight) && Array.isArray(weight.shape)) {
    return weight.shape.reduce((product, value) => product * Number(value || 0), 1);
  }
  return null;
}

function assertLoRAWeightElementCount(label, weight, expected) {
  const actual = getKnownElementCount(weight);
  if (actual === null || actual === expected) return;
  throw new Error(`LoRA ${label} element count mismatch: expected ${expected}, got ${actual}.`);
}

export async function applyLoRA(input, baseOutput, lora, dims, getWeightBuffer, recorder, options = {}) {
  const { M, N, K } = dims;
  const rank = lora.rank;
  const kernelPath = options.kernelPath ?? null;
  if (!rank || rank <= 0) {
    return baseOutput;
  }
  assertLoRAWeightElementCount('A', lora.a, rank * K);
  assertLoRAWeightElementCount('B', lora.b, N * rank);

  const aBuf = getWeightBuffer(lora.a, 'lora_a');
  const bBuf = getWeightBuffer(lora.b, 'lora_b');
  const ownsA = !isGpuBufferInstance(lora.a) && !isWeightBuffer(lora.a);
  const ownsB = !isGpuBufferInstance(lora.b) && !isWeightBuffer(lora.b);
  // Extract underlying GPUBuffer for WeightBuffers
  const aBufGPU = isWeightBuffer(aBuf) ? aBuf.buffer : aBuf;
  const bBufGPU = isWeightBuffer(bBuf) ? bBuf.buffer : bBuf;
  const intermediateDtype = input.dtype;
  const outputDtype = baseOutput.dtype;
  let loraIntermediate = null;
  let loraOutput = null;
  let scaled = null;
  try {
    loraIntermediate = recorder
      ? await recordMatmul(recorder, input, aBuf, M, rank, K, { transposeB: 'auto', role: 'lora_a', kernelPath, outputDtype: intermediateDtype })
      : await runMatmul(input, aBuf, M, rank, K, { transposeB: 'auto', role: 'lora_a', kernelPath, outputDtype: intermediateDtype });

    loraOutput = recorder
      ? await recordMatmul(recorder, loraIntermediate, bBuf, M, N, rank, { transposeB: 'auto', role: 'lora_b', kernelPath, outputDtype })
      : await runMatmul(loraIntermediate, bBuf, M, N, rank, { transposeB: 'auto', role: 'lora_b', kernelPath, outputDtype });

    scaled = recorder
      ? await recordScale(recorder, loraOutput, lora.scale, { outputBuffer: null })
      : await runScale(loraOutput, lora.scale, { outputBuffer: null });

    const combined = recorder
      ? await recordResidualAdd(recorder, baseOutput, scaled, M * N)
      : await runResidualAdd(baseOutput, scaled, M * N);

    if (recorder) {
      recorder.trackTemporaryBuffer(loraIntermediate.buffer);
      recorder.trackTemporaryBuffer(loraOutput.buffer);
      recorder.trackTemporaryBuffer(scaled.buffer);
      if (ownsA) recorder.trackTemporaryBuffer(aBufGPU);
      if (ownsB) recorder.trackTemporaryBuffer(bBufGPU);
    } else {
      releaseBuffer(loraIntermediate.buffer);
      releaseBuffer(loraOutput.buffer);
      releaseBuffer(scaled.buffer);
      if (ownsA) releaseBuffer(aBufGPU);
      if (ownsB) releaseBuffer(bBufGPU);
    }

    return combined;
  } catch (error) {
    if (recorder) {
      if (loraIntermediate) recorder.trackTemporaryBuffer(loraIntermediate.buffer);
      if (loraOutput) recorder.trackTemporaryBuffer(loraOutput.buffer);
      if (scaled) recorder.trackTemporaryBuffer(scaled.buffer);
      if (ownsA) recorder.trackTemporaryBuffer(aBufGPU);
      if (ownsB) recorder.trackTemporaryBuffer(bBufGPU);
    } else {
      if (loraIntermediate) releaseBuffer(loraIntermediate.buffer);
      if (loraOutput) releaseBuffer(loraOutput.buffer);
      if (scaled) releaseBuffer(scaled.buffer);
      if (ownsA) releaseBuffer(aBufGPU);
      if (ownsB) releaseBuffer(bBufGPU);
    }
    throw error;
  }
}
