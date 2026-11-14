import type { CommandRecorder } from '../command-recorder.js';
import type { Tensor } from '../tensor.js';

export interface HyperConnectionGeometry {
  numTokens: number;
  streamCount: number;
  streamWidth: number;
}

export interface HyperConnectionStreamOptions extends HyperConnectionGeometry {
  /** Caller-owned output; must not alias an input. */
  outputBuffer?: GPUBuffer | null;
}

export interface HyperConnectionNormOptions extends HyperConnectionStreamOptions {
  eps: number;
  normWeightOffset: boolean;
}

export type HyperConnectionActivationMode = 'low_rank_silu' | 'mix_sigmoid' | 'inject_gate';

export interface HyperConnectionActivationOptions {
  /** low_rank_silu: silu(x / streamCount); mix_sigmoid: sigmoid(x); inject_gate: 2 * sigmoid(x / streamCount). */
  mode: HyperConnectionActivationMode;
  rows: number;
  columns: number;
  streamCount: number;
  outputBuffer?: GPUBuffer | null;
}

/** Validate stream geometry and return numTokens * streamCount * streamWidth. */
export declare function validateHyperConnectionGeometry(geometry: HyperConnectionGeometry): number;

/** Grouped RMSNorm: [numTokens, streamCount * streamWidth] -> same, (1 + weight) per stream slice. */
export declare function runHyperConnectionNorm(input: Tensor, weight: Tensor, options: HyperConnectionNormOptions): Promise<Tensor>;
export declare function recordHyperConnectionNorm(
  recorder: CommandRecorder, input: Tensor, weight: Tensor, options: HyperConnectionNormOptions
): Promise<Tensor>;

export declare function runHyperConnectionActivation(input: Tensor, options: HyperConnectionActivationOptions): Promise<Tensor>;
export declare function recordHyperConnectionActivation(
  recorder: CommandRecorder, input: Tensor, options: HyperConnectionActivationOptions
): Promise<Tensor>;

/** Branch input: mean over streams of mix * normed -> [numTokens, streamWidth]. */
export declare function runHyperConnectionBranch(mix: Tensor, normed: Tensor, options: HyperConnectionStreamOptions): Promise<Tensor>;
export declare function recordHyperConnectionBranch(
  recorder: CommandRecorder, mix: Tensor, normed: Tensor, options: HyperConnectionStreamOptions
): Promise<Tensor>;

/** Injection: streams + branch * gate[stream] -> [numTokens, streamCount * streamWidth]. */
export declare function runHyperConnectionInject(
  streams: Tensor, branch: Tensor, gate: Tensor, options: HyperConnectionStreamOptions
): Promise<Tensor>;
export declare function recordHyperConnectionInject(
  recorder: CommandRecorder, streams: Tensor, branch: Tensor, gate: Tensor, options: HyperConnectionStreamOptions
): Promise<Tensor>;

/** Entry replication: embeddings [numTokens, streamWidth] -> streams [numTokens, streamCount * streamWidth]. */
export declare function runHyperConnectionReplicate(embeds: Tensor, options: HyperConnectionStreamOptions): Promise<Tensor>;
export declare function recordHyperConnectionReplicate(
  recorder: CommandRecorder, embeds: Tensor, options: HyperConnectionStreamOptions
): Promise<Tensor>;
