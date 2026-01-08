/**
 * LoRA application helpers for matmul outputs.
 *
 * @module inference/pipeline/lora-apply
 */

import type { CommandRecorder } from '../../gpu/command-recorder.js';
import type { Tensor } from '../../gpu/tensor.js';
import type { WeightBuffer } from '../../gpu/weight-buffer.js';
import type { LoRAModuleWeights } from './lora.js';
import type { MaybeGPUBuffer } from './types.js';

interface LoRADims {
  M: number;
  N: number;
  K: number;
}

export function applyLoRA(
  input: Tensor,
  baseOutput: Tensor,
  lora: LoRAModuleWeights,
  dims: LoRADims,
  getWeightBuffer: (weight: MaybeGPUBuffer, label: string) => GPUBuffer | WeightBuffer,
  recorder?: CommandRecorder
): Promise<Tensor>;
