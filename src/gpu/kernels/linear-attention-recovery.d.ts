import type { CommandRecorder } from '../command-recorder.js';
import type { SpeculativeAcceptance } from './speculative-accept.js';
import type { LinearAttentionCoreLayerState } from './linear-attention-core.js';

export interface RecoveryEntry {
  copies: Array<{source: GPUBuffer; saved: GPUBuffer}>;
  recover(recorder: CommandRecorder, acceptance: SpeculativeAcceptance): Promise<void>;
}
export interface LinearAttentionRecovery {
  readonly layerCount: number;
  /** Caller retains parent recorder cleanup until every recovery submit has settled. */
  recordRecover(recorder: CommandRecorder, acceptance: SpeculativeAcceptance): Promise<void>;
  /** Restore initial GPU state after failure/cancellation; caller also restores CPU cursors and KV length. */
  recordRestore(recorder: CommandRecorder): void;
  /** Ends capture/borrowing. Does not release resources ahead of the parent recorder. */
  close(): void;
}
/** Explicitly capture a single verification window's core projection inputs and initial GPU states. */
export function createLinearAttentionRecovery(recorder: CommandRecorder, numTokens: number): LinearAttentionRecovery;
/** Internal hook for the existing core owner; never uses observation probes to modify execution. */
export function captureLinearAttentionRecovery(
  recorder: CommandRecorder, state: LinearAttentionCoreLayerState, numTokens: number,
  recover: RecoveryEntry['recover']
): void;

/** Ordinary execution does not capture state or allocate recovery closures. */
export function isLinearAttentionRecoveryRecording(recorder: CommandRecorder): boolean;
