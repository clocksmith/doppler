import type { PipelineState } from './state.js';
import type { AnyLayerPartitionPlan } from './layer-partition-contract.js';
import type { GpuLogitsResult } from './generator/token-selection.js';
import type { PartitionTiming } from './resident-partition-contract.js';
export type PartitionExecutionState = PipelineState & {
  visionCapable?: boolean; audioCapable?: boolean; operatorDiagnostics?: unknown;
  modelPartition?: { plan: AnyLayerPartitionPlan; index: number } | null;
};
export function assertPartitionExecutionSupported(state: PartitionExecutionState, plan: AnyLayerPartitionPlan): void;
export type ExecutePartitionChunk = (state: PartitionExecutionState,
  input: { numTokens: number; tokenIds?: number[]; activationBytes?: ArrayBuffer | ArrayBufferView },
  signal: AbortSignal, prefill: boolean, finalChunk: boolean) => Promise<{
    activationBytes?: ArrayBuffer; logits?: GpuLogitsResult; timing: PartitionTiming;
  }>;
export function executePartitionLayers(state: PartitionExecutionState,
  input: { numTokens: number; tokenIds?: number[]; activationBytes?: ArrayBuffer | ArrayBufferView },
  signal: AbortSignal): Promise<({ activationBytes: ArrayBuffer; logits?: never } | { logits: GpuLogitsResult; activationBytes?: never }) & {timing: PartitionTiming}>;
