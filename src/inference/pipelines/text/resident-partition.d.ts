import type { PartitionExecutionState } from './partition-execution.js';
import type { PartitionAttempt } from './partition-attempt.js';
import type { ResidentPartitionAllocation, ResidentExecutionIdentity, ResidentPartitionSession,
  ResidentPartitionTokenPorts } from './resident-partition-contract.js';
import type { IncrementalTokenDecoder } from '../../tokenizers/bundled/incremental-decoder.js';
export interface ResidentAttempt {
  binding: string; identity: ResidentExecutionIdentity; nonce: string;
  step: number; position: number; retired: boolean; done: boolean;
  controller: AbortController; pending: Promise<unknown> | null; settlement: Promise<void> | null;
  execution: PartitionAttempt | null; generationDigest: string | null;
  contextTokens: number[]; text: string; decoder: IncrementalTokenDecoder | null;
  canonicalState: import('../../resident-target-state.js').ResidentTargetState | null;
  canonicalMirror: import('../../resident-target-state.js').ResidentCursorMirror | null;
  canonicalPrompt: number[] | null; canonicalAwaiting: boolean;
}
/** Takes ownership of the loaded pipeline through the supplied close port. */
export function createResidentPartitionSession(pipeline: PartitionExecutionState,
  allocation: ResidentPartitionAllocation, tokens: ResidentPartitionTokenPorts,
  closeProgram: () => Promise<void>, signal?: AbortSignal,
  onLoadProgress?: ((progress: import('./resident-partition-contract.js').ResidentPipelinePreparationProgress) => void) | null
): Promise<ResidentPartitionSession>;
