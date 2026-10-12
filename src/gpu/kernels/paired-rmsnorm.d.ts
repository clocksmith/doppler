import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { WeightBuffer, TensorLike } from '../weight-buffer.js';
import type { RMSNormOptions } from './rmsnorm.js';
/** Registered paired RMSNorm execution, preserving source epsilon in two words. */
export declare function executePairedRMSNorm(recorder: CommandRecorder | null, input: Tensor,
  weight: GPUBuffer | WeightBuffer | TensorLike, eps: number,
  options: RMSNormOptions & { batchSize: number; hiddenSize: number }): Promise<Tensor>;
