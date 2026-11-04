import type { Tensor } from '../../tensor.js';
import type { CommandRecorder } from '../../command-recorder.js';

export interface QuerySplitKOptions {
  seqLen: number;
  kvLen: number;
  numKVHeads: number;
  scale: number;
  causal: boolean;
  startPos: number;
  outputBuffer: GPUBuffer | null;
  attnSoftcap: number;
  slidingWindow: number | null;
  kvLenBuffer: GPUBuffer | null;
  kvStart: number;
  kvLayout: string | null;
  kvPageTable: GPUBuffer | null;
  kvPageSize: number;
  bidirectionalSpanLength: number;
}

/** Execute the declared producer/reducer pair for contiguous causal scalar F32 attention. */
export declare function executeQuerySplitK(recorder: CommandRecorder | null, variant: string,
  Q: Tensor, K: Tensor, V: Tensor, numHeads: number, headDim: number,
  options: QuerySplitKOptions): Promise<Tensor>;
