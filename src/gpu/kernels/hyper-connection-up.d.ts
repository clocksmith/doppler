import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { HyperConnectionStreamOptions } from './hyper-connection.js';

/** Single-token, four-stream Q4_0 gemv32 program; unsupported storage fails without fallback. */
export interface HyperConnectionUpMixBranchOptions extends HyperConnectionStreamOptions {
  projectionRank: number;
  signal?: AbortSignal | null;
}
export declare function runHyperConnectionUpMixBranch(
  input: Tensor, weights: unknown, normed: Tensor, options: HyperConnectionUpMixBranchOptions
): Promise<Tensor>;
export declare function recordHyperConnectionUpMixBranch(
  recorder: CommandRecorder, input: Tensor, weights: unknown, normed: Tensor, options: HyperConnectionUpMixBranchOptions
): Promise<Tensor>;
