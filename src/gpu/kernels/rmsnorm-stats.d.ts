import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';

export interface RMSNormStatsOptions {
  batchSize?: number;
  hiddenSize: number;
  outputBuffer?: GPUBuffer | null;
  label?: string;
}

export interface RMSNormStatsResult {
  prenormSum: Tensor;
  invRmsBuffer: GPUBuffer;
}

export declare function runRMSNormStats(
  input: Tensor,
  residual: Tensor,
  eps: number,
  options?: RMSNormStatsOptions
): Promise<RMSNormStatsResult>;

export declare function recordRMSNormStats(
  recorder: CommandRecorder,
  input: Tensor,
  residual: Tensor,
  eps: number,
  options?: RMSNormStatsOptions
): Promise<RMSNormStatsResult>;
