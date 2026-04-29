import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';

export interface LinearAttentionOptions extends OutputBufferOptions {
  numHeads: number;
  headDim: number;
  numTokens?: number;
  hiddenSize?: number;
  eps?: number;
  summaryBuffer?: GPUBuffer | null;
}

export declare function runLinearAttention(
  query: Tensor,
  key: Tensor,
  value: Tensor,
  options: LinearAttentionOptions
): Promise<Tensor>;

export declare function recordLinearAttention(
  recorder: CommandRecorder,
  query: Tensor,
  key: Tensor,
  value: Tensor,
  options: LinearAttentionOptions
): Promise<Tensor>;
