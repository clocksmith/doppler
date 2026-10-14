import type { Tensor } from '../tensor.js';
import type { KernelPipelineRequest } from './pipeline-cache.js';
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

/** Bias, scale and softcap are uniforms; only the input dtype selects the program. */
export declare function resolveLogitFinalizePipeline(inputDtype: 'f16' | 'f32'): KernelPipelineRequest;
