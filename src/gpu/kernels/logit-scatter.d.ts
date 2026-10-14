import type { Tensor } from '../tensor.js';
import type { KernelPipelineRequest } from './pipeline-cache.js';

export interface LogitScatterOptions {
  rowCount: number;
  chunkColumns: number;
  targetColumns: number;
  columnOffset: number;
}

export declare function runLogitScatter(
  input: Tensor,
  outputBuffer: GPUBuffer,
  options: LogitScatterOptions
): Promise<Tensor>;

/** Chunk geometry is uniform; only the input dtype selects the program. */
export declare function resolveLogitScatterPipeline(inputDtype: 'f16' | 'f32'): KernelPipelineRequest;
