import type { PartitionExecutionState } from './partition-execution.js';
import type { ResidentPartitionAllocation } from './resident-partition-contract.js';

export interface ResidentPartitionWarmupReceipt {
  promptTokens: number;
  decodeSteps: number;
  /** false when the stage starts from a boundary activation and ran zeros of the declared shape. */
  tokenInput: boolean;
  durationMs: number;
}

/** Runs runtime.shared.kernelWarmup.warmPartition in a private attempt; null when it is not configured. */
export function runResidentPartitionWarmup(owner: PartitionExecutionState, allocation: ResidentPartitionAllocation,
  signal: AbortSignal | null | undefined): Promise<ResidentPartitionWarmupReceipt | null>;
