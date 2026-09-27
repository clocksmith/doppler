import type { PartitionExecutionState } from './partition-execution.js';
/** The enclosing operation must settle submitted work before closing its attempt. */
export interface PartitionAttempt { state: PartitionExecutionState; close(): void; }
export function createPartitionAttempt(owner: PartitionExecutionState): PartitionAttempt;
