/**
 * LoRA application helpers for matmul outputs.
 *
 * @module inference/pipeline/lora-apply
 */

import { releaseBuffer } from '../../gpu/buffer-pool.js';
import type { CommandRecorder } from '../../gpu/command-recorder.js';
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
  input: GPUBuffer,
  baseOutput: GPUBuffer,
  lora: LoRAModuleWeights,
  dims: LoRADims,
  getWeightBuffer: (weight: MaybeGPUBuffer, label: string) => GPUBuffer,
  recorder?: CommandRecorder
): Promise<GPUBuffer> {
  const { M, N, K } = dims;
  const rank = lora.rank;
  if (!rank || rank <= 0) {
    return baseOutput;
  }

  const aBuf = getWeightBuffer(lora.a, 'lora_a');
  const bBuf = getWeightBuffer(lora.b, 'lora_b');
  const ownsA = !(lora.a instanceof GPUBuffer);
  const ownsB = !(lora.b instanceof GPUBuffer);

  const loraIntermediate = recorder
    ? await recordMatmul(recorder, input, aBuf, M, rank, K, { transposeB: 'auto' })
    : await runMatmul(input, aBuf, M, rank, K, { transposeB: 'auto' });

  const loraOutput = recorder
    ? await recordMatmul(recorder, loraIntermediate, bBuf, M, N, rank, { transposeB: 'auto' })
    : await runMatmul(loraIntermediate, bBuf, M, N, rank, { transposeB: 'auto' });

  const scaled = recorder
    ? await recordScale(recorder, loraOutput, lora.scale, { outputBuffer: null })
    : await runScale(loraOutput, lora.scale, { outputBuffer: null });

  const combined = recorder
    ? await recordResidualAdd(recorder, baseOutput, scaled, M * N)
    : await runResidualAdd(baseOutput, scaled, M * N);

  if (recorder) {
    recorder.trackTemporaryBuffer(loraIntermediate);
    recorder.trackTemporaryBuffer(loraOutput);
    recorder.trackTemporaryBuffer(scaled);
    if (ownsA) recorder.trackTemporaryBuffer(aBuf);
    if (ownsB) recorder.trackTemporaryBuffer(bBuf);
  } else {
    releaseBuffer(loraIntermediate);
    releaseBuffer(loraOutput);
    releaseBuffer(scaled);
    if (ownsA) releaseBuffer(aBuf);
    if (ownsB) releaseBuffer(bBuf);
  }

  return combined;
}
