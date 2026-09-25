/** Partition metadata helpers; these do not execute or qualify distributed inference. */
export declare const LAYER_PARTITION_SCHEMA: 'doppler.layer-partition-contract/v1';
export declare const ACTIVATION_TENSOR_SCHEMA: 'doppler.activation-tensor/v1';
export declare const PARTITION_COMPARISON_SCHEMA: 'doppler.partition-comparison-contract/v1';
export declare const DEFAULT_NUMERICAL_TOLERANCE: number;
export declare const DEFAULT_COSINE_SIMILARITY_MIN: number;
export type ActivationDtype = 'f32' | 'f16';
export interface PartitionTensorContract {
  type: 'token-ids' | 'activation-tensor' | 'logits';
  rank: number;
  shapeDescription: string;
  dtype: ActivationDtype | 'i32';
  hiddenSize?: number;
  vocabSize?: number;
}
export interface LayerPartition {
  index: number;
  layerRange: number[];
  layerCount: number;
  hasEmbedding: boolean;
  hasLmHead: boolean;
  inputContract: PartitionTensorContract;
  outputContract: PartitionTensorContract;
}
export interface LayerPartitionPlan {
  readonly schema: typeof LAYER_PARTITION_SCHEMA;
  readonly modelId: string;
  readonly totalLayers: number;
  readonly hiddenSize: number;
  readonly vocabSize: number;
  readonly splitLayer: number;
  readonly activationDtype: ActivationDtype;
  readonly partitions: readonly Readonly<LayerPartition>[];
}
export declare function createLayerPartitionPlan(options: {
  modelId: string; numLayers: number; hiddenSize: number; vocabSize: number;
  splitLayer?: number | null; activationDtype?: ActivationDtype;
}): LayerPartitionPlan;
export declare function validateActivationTensorShape(options: {
  shape: readonly number[]; dtype: ActivationDtype | 'i32';
  byteLength?: number | null; hiddenSize?: number | null;
}): { batchSize: number; seqLen: number; hiddenSize: number; expectedBytes: number };
export interface ActivationFrame {
  readonly schema: typeof ACTIVATION_TENSOR_SCHEMA;
  readonly shape: readonly number[];
  readonly dtype: ActivationDtype;
  readonly seqOffset: number;
  readonly step: number;
  readonly byteLength: number;
  readonly metadata: Readonly<Record<string, unknown>>;
  readonly buffer: ArrayBuffer;
}
export declare function serializeActivationFrame(options: {
  shape: readonly number[]; dtype?: ActivationDtype; data: ArrayBuffer | ArrayBufferView;
  seqOffset?: number; step?: number; metadata?: Record<string, unknown>;
}): ActivationFrame;
export declare function deserializeActivationFrame(frame: ActivationFrame): {
  shape: number[]; dtype: ActivationDtype; seqOffset: number; step: number;
  tensorData: Float32Array | Uint16Array; metadata: Readonly<Record<string, unknown>>;
};
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
