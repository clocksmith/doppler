import type { CommandRecorder } from '../../../../gpu/command-recorder.js';
import type { Tensor } from '../../../../gpu/tensor.js';
import type { ResolvedNgramEmbedding } from '../../../../config/schema/ngram-embedding.schema.js';
import type { ResolvedHyperConnections } from '../../../../config/schema/hyper-connection-decoder.schema.js';

export declare class PleConvState {
  constructor(options: { channels: number; stateLength: number });
  readonly channels: number;
  readonly stateLength: number;
  readonly epoch: number;
  readonly destroyed: boolean;
  readonly buffer: GPUBuffer;
  /** Zero the state for a new sequence; advances the epoch. */
  reset(): void;
  destroy(): Promise<void>;
}

export interface PleLayerWeights {
  keyProj: unknown;
  valueProj: unknown;
  normKey: Tensor;
  normQuery: Tensor;
  normConv: Tensor;
  /** conv1d weight flattened to [channels * kernel] f32. */
  conv: Tensor;
}

/** Caller-owned projection (kernel choice and weight dtype stay with the execution plan). */
export type PleProjection = (
  role: 'ple_key_proj' | 'ple_value_proj',
  input: Tensor,
  weight: unknown,
  outFeatures: number,
  inFeatures: number
) => Promise<Tensor>;

export interface PleLayerInput {
  /** Residual streams [numTokens, streamCount * streamWidth]. */
  streams: Tensor;
  /** Gathered BF16 table rows [numTokens, embedDim] in address order. */
  rows: GPUBuffer;
  weights: PleLayerWeights;
  state: PleConvState;
  ngram: ResolvedNgramEmbedding;
  hyper: ResolvedHyperConnections;
  numTokens: number;
  project: PleProjection;
}

/** New residual streams and the signed-square-root gate before the sigmoid. Caller owns both. */
export declare function runPleLayer(
  target: CommandRecorder | null,
  input: PleLayerInput
): Promise<{ streams: Tensor; gateInput: Tensor }>;
