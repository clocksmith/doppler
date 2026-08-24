import type { Tensor } from '../tensor.js';

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
