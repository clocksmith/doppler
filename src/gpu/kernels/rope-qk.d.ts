import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { RoPEOptions } from './rope.js';
import type { KernelPipelineRequest } from './pipeline-cache.js';

export interface RoPEQKOptions extends Omit<RoPEOptions, 'numHeads'> {
  numQHeads: number;
  numKVHeads: number;
  headDim: number;
  rotaryDim?: number;
  pairSpanDim?: number;
}

export interface RoPEQKResult {
  q: Tensor;
  k: Tensor;
}

/** The joint Q/K RoPE program for a component dtype and precision; positions and lengths are uniforms. */
export function resolveRoPEQKPipeline(
  dtype: 'f16' | 'f32',
  precision: 'two-f32' | undefined
): KernelPipelineRequest;

export function canUseRoPEQK(
  qTensor: Tensor | null | undefined,
  kTensor: Tensor | null | undefined,
  options?: { reusesSharedKV?: boolean }
): boolean;

export declare function runRoPEQK(
  qTensor: Tensor,
  kTensor: Tensor,
  freqsCos: GPUBuffer | Tensor,
  freqsSin: GPUBuffer | Tensor,
  seqLen: number,
  options: RoPEQKOptions
): Promise<RoPEQKResult>;

export declare function recordRoPEQK(
  recorder: CommandRecorder,
  qTensor: Tensor,
  kTensor: Tensor,
  freqsCos: GPUBuffer | Tensor,
  freqsSin: GPUBuffer | Tensor,
  seqLen: number,
  options: RoPEQKOptions
): Promise<RoPEQKResult>;
