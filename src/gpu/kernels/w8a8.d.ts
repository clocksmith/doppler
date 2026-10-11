import type { Tensor } from '../tensor.js';
import type { WeightBuffer } from '../weight-buffer.js';
import type { CommandRecorder } from '../command-recorder.js';
export interface W8A8Options {
  variant: 'w8a8_portable' | 'w8a8_packed';
  outputDtype: 'f32';
  outputBuffer?: GPUBuffer | null;
  signal?: AbortSignal;
  alpha?: number;
  transposeB?: boolean | 'auto';
  aOffset?: number;
  bOffset?: number;
  cOffset?: number;
  constants?: Record<string, number | boolean>;
}
export type W8A8Uniforms = { rows: number; columns: number; inner: number; row_words: number };
export function assertW8A8Status(status: GPUBuffer): Promise<void>;
export function validateW8A8Matmul(device: GPUDevice, input: Tensor, weights: WeightBuffer,
  M: number, N: number, K: number, options: W8A8Options): W8A8Uniforms;
export function runW8A8Matmul(input: Tensor, weights: WeightBuffer,
  M: number, N: number, K: number, options: W8A8Options): Promise<Tensor>;
export function recordW8A8Matmul(recorder: CommandRecorder, input: Tensor, weights: WeightBuffer,
  M: number, N: number, K: number, options: W8A8Options): Promise<Tensor>;
