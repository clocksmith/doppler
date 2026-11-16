import type { PrefillPolicyState } from '../generator/prefill/prefill-policy.js';
import type { PipelineState } from "../state.js";
import type { AnyLayerPartitionPlan } from "../execution/layer-partition-contract.js";
import type { GpuLogitsResult } from "../generator/sampling/token-selection.js";
import type { PartitionTiming } from './resident-partition-contract.js';
export type PartitionExecutionState = PipelineState & {
  visionCapable?: boolean; audioCapable?: boolean; operatorDiagnostics?: unknown;
  modelPartition?: { plan: AnyLayerPartitionPlan; index: number } | null;
};
export function assertPartitionExecutionSupported(state: PartitionExecutionState, plan: AnyLayerPartitionPlan, requireAdapter?: boolean): void;
export type ExecutePartitionChunk = (state: PartitionExecutionState,
  input: { numTokens: number; tokenIds?: number[]; activationBytes?: ArrayBuffer | ArrayBufferView },
  signal: AbortSignal, prefill: boolean, finalChunk: boolean) => Promise<{
    activationBytes?: ArrayBuffer; logits?: GpuLogitsResult; timing: PartitionTiming;
  }>;
export function executePartitionLayers(state: PartitionExecutionState,
  input: { numTokens: number; tokenIds?: number[]; activationBytes?: ArrayBuffer | ArrayBufferView },
  signal: AbortSignal): Promise<({ activationBytes: ArrayBuffer; logits?: never } | { logits: GpuLogitsResult; activationBytes?: never }) & {timing: PartitionTiming}>;

/** Preserve the owner's complete loader/cache policy while selecting the phase's recording mode. */
export function shouldRecordPartitionLayers(state: PrefillPolicyState & { currentSeqLen: number },
  disableCommandBatching: boolean, prefill: boolean, finalChunk: boolean): boolean;
