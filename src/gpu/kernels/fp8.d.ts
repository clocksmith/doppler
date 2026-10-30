import type { Tensor } from '../tensor.js';
import type { WeightBuffer } from '../weight-buffer.js';
import type { CommandRecorder } from '../command-recorder.js';
export interface FP8MatmulOptions {
  variant?: string | null;
  outputDtype: 'f32';
  outputBuffer?: GPUBuffer | null;
  transposeB?: boolean | 'auto';
  alpha?: number;
  aOffset?: number;
  bOffset?: number;
  cOffset?: number;
  constants?: Record<string, number | boolean>;
  signal?: AbortSignal;
}
export function runFP8Matmul(input: Tensor, weights: WeightBuffer, M: number, N: number, K: number,
  options: FP8MatmulOptions): Promise<Tensor>;
export function recordFP8Matmul(recorder: CommandRecorder, input: Tensor, weights: WeightBuffer,
  M: number, N: number, K: number, options: FP8MatmulOptions): Promise<Tensor>;
