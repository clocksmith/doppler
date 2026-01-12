/**
 * check-stop.ts - GPU Stop Condition Kernel
 *
 * Checks if generation should stop based on EOS token or max length.
 * Used in GPU-only decode loop to eliminate CPU roundtrips.
 */

import type { CommandRecorder } from '../command-recorder.js';

export interface CheckStopParams {
  sampledTokenBuffer: GPUBuffer;  // u32 storage buffer with sampled tokens
  tokenIndex?: number;
  eosTokenId: number;
  maxTokens: number;
  currentPos: number;
}

/**
 * Record a stop condition check into the command buffer.
 * Returns the shouldStop buffer (1 if should stop, 0 otherwise).
 */
export declare function recordCheckStop(
  recorder: CommandRecorder,
  params: CheckStopParams
): GPUBuffer;

/**
 * Standalone check stop (submits immediately, for testing).
 */
export declare function checkStop(params: CheckStopParams): Promise<boolean>;
