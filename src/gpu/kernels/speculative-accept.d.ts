import type { CommandRecorder } from '../command-recorder.js';

/** Compact result: [emittedCount, acceptedDrafts, status, lastInputIndex, nextToken, ...targetIDs]. */
export interface SpeculativeAcceptance {
  readonly buffer: GPUBuffer;
  readonly maxTokens: number;
}
export interface SpeculativeAcceptanceOptions {
  draftCount: number;
  outputLimit: number;
  vocabSize: number;
  /** Caller-owned, distinct STORAGE | COPY_SRC buffer, at least (draftCount + 6) * 4 bytes. */
  outputBuffer: GPUBuffer;
}
/** Record greedy prefix acceptance only; stochastic probability-ratio sampling is unsupported here. */
export function recordSpeculativeAcceptance(
  recorder: CommandRecorder, drafts: GPUBuffer, targets: GPUBuffer, options: SpeculativeAcceptanceOptions
): Promise<SpeculativeAcceptance>;
/** GPU-resident cap must be in [1, outputLimit]; an invalid cap yields failed status and zero updates.
 * Enables a captured window to retain its dispatch geometry for final-token or prefix recovery limits.
 */
export function recordSpeculativeAcceptanceFromGPU(
  recorder: CommandRecorder, drafts: GPUBuffer, targets: GPUBuffer, outputCap: GPUBuffer,
  options: SpeculativeAcceptanceOptions
): Promise<SpeculativeAcceptance>;
/** Only accepts an unforgeable receipt from recordSpeculativeAcceptance on this device. */
export function createSpeculativeRecoveryUniform(
  recorder: CommandRecorder, acceptance: SpeculativeAcceptance, payload: ArrayBuffer, maxTokens: number
): GPUBuffer;

/** Record accepted last hidden row and next token for the next window, without a CPU index or extra submission.
 * Caller retains/discards these tentative carried tensors with the verification attempt and checks result status.
 */
export function recordSpeculativeCarry(
  recorder: CommandRecorder, acceptance: SpeculativeAcceptance, hidden: import('../tensor.js').Tensor,
  hiddenSize: number, carriedHidden: GPUBuffer, carriedToken: GPUBuffer
): Promise<void>;
