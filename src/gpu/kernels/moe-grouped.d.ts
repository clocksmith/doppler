import type { Tensor } from '../tensor.js';
import type { WeightBuffer } from '../weight-buffer.js';
import type { CommandRecorder } from '../command-recorder.js';
export interface GroupedPrefillGeometry {
  /** Exact declared program; omission preserves the original direct component API. */
  kernelPath?: import('../../config/schema/index.js').KernelPathSchema | null;
  phase?: 'prefill' | 'decode';
  layerIdx?: number;
  numTokens: number;
  numExperts: number;
  topK: number;
  hiddenSize: number;
  intermediateSize: number;
}
/** Admission is fail closed. No readback or automatic algorithm fallback. */
export declare function groupedPrefillGeometry(options: GroupedPrefillGeometry, limits: GPUSupportedLimits): {
  numRoutes: number; maxChunks: number; pairOffset: number; groupBytes: number; guGroups: number; downGroups: number;
};
/** Indices must be the GPU router's distinct, valid top-k IDs in token/slot order. */
export declare function executeGroupedSort(recorder: CommandRecorder | null, indices: GPUBuffer,
  groups: GPUBuffer, indirect: GPUBuffer, options: GroupedPrefillGeometry): Promise<void>;
/** Returns route-major projections. Grouping never changes route slot identity. */
export declare function executeGroupedProjection(recorder: CommandRecorder | null,
  operation: 'moe_grouped_gu' | 'moe_grouped_down', input: Tensor, weights: WeightBuffer,
  upWeights: WeightBuffer | null, groups: GPUBuffer, indirect: GPUBuffer,
  options: GroupedPrefillGeometry): Promise<Tensor>;
