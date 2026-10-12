import type { PartitionAttempt } from './partition-attempt.js';
export type PartitionReplayInputs = {
  tokenIds: readonly number[]; boundaryRows: readonly Uint8Array[];
} | { activationRows: readonly Uint8Array[] };
/** Caller pins the immutable parent, admits child/host retention, and supplies
 * only the accepted predecessor rows. No target selection or decoding occurs. */
export function replayPartitionAttempt(parent: PartitionAttempt, inputs: PartitionReplayInputs,
  signal: AbortSignal): Promise<PartitionAttempt>;
