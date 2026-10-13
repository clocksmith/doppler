import type { ActivationLayout, ActivationDtype, LayerPartitionPlan, OrderedLayerPartitionPlan } from '../../../config/layer-partition-plan.js';
import { ACTIVATION_TENSOR_SCHEMA, PAIRED_ACTIVATION_TENSOR_SCHEMA, PARTITION_COMPARISON_SCHEMA } from '../../../config/layer-partition-plan.js';
export * from '../../../config/layer-partition-plan.js';
export declare function validateActivationTensorShape(options: ActivationLayout & {
  shape: readonly number[]; dtype: ActivationDtype | 'i32';
  byteLength?: number | null; hiddenSize?: number | null;
}): { batchSize: number; seqLen: number; hiddenSize: number; expectedBytes: number };
export interface ActivationFrame extends ActivationLayout {
  readonly schema: typeof ACTIVATION_TENSOR_SCHEMA | typeof PAIRED_ACTIVATION_TENSOR_SCHEMA;
  readonly shape: readonly number[];
  readonly dtype: ActivationDtype;
  readonly seqOffset: number;
  readonly step: number;
  readonly byteLength: number;
  readonly metadata: Readonly<Record<string, unknown>>;
  readonly buffer: ArrayBuffer;
}
export declare function serializeActivationFrame(options: ActivationLayout & {
  shape: readonly number[]; dtype?: ActivationDtype; data: ArrayBuffer | ArrayBufferView;
  seqOffset?: number; step?: number; metadata?: Record<string, unknown>;
}): ActivationFrame;
export declare function deserializeActivationFrame(frame: ActivationFrame): ActivationLayout & {
  shape: number[]; dtype: ActivationDtype; seqOffset: number; step: number;
  tensorData: Float32Array | Uint16Array; metadata: Readonly<Record<string, unknown>>;
};
export declare function validateActivationFrameForPlan(frame: ActivationFrame, plan: LayerPartitionPlan | OrderedLayerPartitionPlan): ReturnType<typeof deserializeActivationFrame>;
export interface PartitionContinuation {
  readonly partitionIndex: number;
  readonly layerRange: readonly number[];
  getSequenceOffset(): number;
  advance(steps?: number): number;
  getLayerKVCache(layerIdx: number): unknown;
  setLayerKVCache(layerIdx: number, state: unknown): void;
  reset(): void;
}
export declare function createPartitionContinuation(options: {
  partitionIndex: number; totalLayers: number; layerRange: readonly number[]; seqOffset?: number;
}): PartitionContinuation;
export declare function comparePartitionExecution(options: {
  splitOutput: Float32Array | readonly number[];
  referenceOutput: Float32Array | readonly number[];
  tolerance?: number; minCosineSimilarity?: number;
}): Readonly<{
  schema: typeof PARTITION_COMPARISON_SCHEMA; matches: boolean; maxDiff: number;
  mse: number; cosineSimilarity: number; tolerance: number;
  minCosineSimilarity: number; length: number;
}>;
