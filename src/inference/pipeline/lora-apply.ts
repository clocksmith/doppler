/**
 * LoRA application helpers for matmul outputs.
 *
 * @module inference/pipeline/lora-apply
 */

import { releaseBuffer } from '../../gpu/buffer-pool.js';
import type { CommandRecorder } from '../../gpu/command-recorder.js';
import { createTensor, type Tensor } from '../../gpu/tensor.js';
import { type WeightBuffer, isWeightBuffer } from '../../gpu/weight-buffer.js';
import { runMatmul, recordMatmul } from '../../gpu/kernel-selector.js';
import { runResidualAdd, recordResidualAdd } from '../../gpu/kernels/residual.js';
import { runScale, recordScale } from '../../gpu/kernels/scale.js';
import type { LoRAModuleWeights } from './lora.js';
import type { MaybeGPUBuffer } from './types.js';

interface LoRADims {
  M: number;
  N: number;
  K: number;
}

export async function applyLoRA(
  input: Tensor,
  baseOutput: Tensor,
  lora: LoRAModuleWeights,
  dims: LoRADims,
  getWeightBuffer: (weight: MaybeGPUBuffer, label: string) => GPUBuffer | WeightBuffer,
  recorder?: CommandRecorder
): Promise<Tensor> {
  const { M, N, K } = dims;
  const rank = lora.rank;
  if (!rank || rank <= 0) {
    return baseOutput;
  }

  const aBuf = getWeightBuffer(lora.a, 'lora_a');
  const bBuf = getWeightBuffer(lora.b, 'lora_b');
  const ownsA = !(lora.a instanceof GPUBuffer) && !isWeightBuffer(lora.a);
  const ownsB = !(lora.b instanceof GPUBuffer) && !isWeightBuffer(lora.b);

  const loraIntermediate = recorder
    ? await recordMatmul(recorder, input, aBuf, M, rank, K, { transposeB: 'auto', role: 'lora_a' })
    : await runMatmul(input, aBuf, M, rank, K, { transposeB: 'auto', role: 'lora_a' });

  const loraOutput = recorder
    ? await recordMatmul(recorder, loraIntermediate, bBuf, M, N, rank, { transposeB: 'auto', role: 'lora_b' })
    : await runMatmul(loraIntermediate, bBuf, M, N, rank, { transposeB: 'auto', role: 'lora_b' });

  const scaled = recorder
    ? await recordScale(recorder, loraOutput, lora.scale, { outputBuffer: null })
    : await runScale(loraOutput, lora.scale, { outputBuffer: null });

  const combined = recorder
    ? await recordResidualAdd(recorder, baseOutput, scaled, M * N)
    : await runResidualAdd(baseOutput, scaled, M * N);

  // Extract underlying GPUBuffer for WeightBuffers
  const aBufGPU = isWeightBuffer(aBuf) ? aBuf.buffer : aBuf;
  const bBufGPU = isWeightBuffer(bBuf) ? bBuf.buffer : bBuf;

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
}
