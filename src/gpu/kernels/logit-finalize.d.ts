import type { Tensor } from '../tensor.js';
import type { WeightBuffer } from '../weight-buffer.js';
import type { CommandRecorder } from '../command-recorder.js';

export interface FinalizeLogitsTensorOptions {
  rowCount: number;
  sourceColumns: number;
  targetColumns: number;
  bias?: Float32Array | GPUBuffer | WeightBuffer | null;
  outputScale: number;
  softcap: number;
}

export declare function runFinalizeLogitsTensor(
  input: Tensor,
  options: FinalizeLogitsTensorOptions
): Promise<Tensor>;

export declare function recordFinalizeLogitsTensor(
  recorder: CommandRecorder,
  input: Tensor,
  options: FinalizeLogitsTensorOptions
): Promise<Tensor>;
