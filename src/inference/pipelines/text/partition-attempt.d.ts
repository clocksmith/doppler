import type { PartitionExecutionState } from './partition-execution.js';
/** The enclosing operation must settle submitted work before closing its attempt. */
export interface PartitionAttempt { state: PartitionExecutionState; close(): void; }
export function createPartitionAttempt(owner: PartitionExecutionState, maxSeqLen: number): PartitionAttempt;
/** Parent must remain pinned until this operation settles. */
export function clonePartitionAttempt(parent: PartitionAttempt, signal: AbortSignal): Promise<PartitionAttempt>;
