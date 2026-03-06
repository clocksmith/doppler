import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';

export interface SanaLinearAttentionOptions extends OutputBufferOptions {
  numHeads: number;
  headDim: number;
  numTokens?: number;
  hiddenSize?: number;
  eps?: number;
  summaryBuffer?: GPUBuffer | null;
}

export declare function runSanaLinearAttention(
  query: Tensor,
  key: Tensor,
  value: Tensor,
  options: SanaLinearAttentionOptions
): Promise<Tensor>;

export declare function recordSanaLinearAttention(
  recorder: CommandRecorder,
  query: Tensor,
  key: Tensor,
  value: Tensor,
  options: SanaLinearAttentionOptions
): Promise<Tensor>;
