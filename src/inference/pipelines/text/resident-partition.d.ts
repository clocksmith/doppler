import type { PartitionExecutionState } from './partition-execution.js';
import type { PartitionAttempt } from './partition-attempt.js';
import type { ResidentPartitionAllocation, ResidentPartitionIdentity, ResidentPartitionSession,
  ResidentPartitionTokenPorts } from './resident-partition-contract.js';
import type { IncrementalTokenDecoder } from '../../tokenizers/bundled/incremental-decoder.js';
export interface ResidentAttempt {
  binding: string; identity: ResidentPartitionIdentity; nonce: string;
  step: number; position: number; retired: boolean; done: boolean;
  controller: AbortController; pending: Promise<unknown> | null; settlement: Promise<void> | null;
  execution: PartitionAttempt | null; generationDigest: string | null;
  contextTokens: number[]; text: string; decoder: IncrementalTokenDecoder | null;
}
/** Takes ownership of the loaded pipeline through the supplied close port. */
export function createResidentPartitionSession(pipeline: PartitionExecutionState,
  allocation: ResidentPartitionAllocation, tokens: ResidentPartitionTokenPorts,
  closeProgram: () => Promise<void>): Promise<ResidentPartitionSession>;
