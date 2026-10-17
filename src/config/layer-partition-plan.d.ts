/** Partition metadata helpers; these do not execute or qualify distributed inference. */
export declare const LAYER_PARTITION_SCHEMA: 'doppler.layer-partition-contract/v1';
export declare const ACTIVATION_TENSOR_SCHEMA: 'doppler.activation-tensor/v1';
export declare const ORDERED_LAYER_PARTITION_SCHEMA: 'doppler.layer-partition-contract/v3';
export declare const PAIRED_LAYER_PARTITION_SCHEMA: 'doppler.layer-partition-contract/v2';
export declare const PAIRED_ACTIVATION_TENSOR_SCHEMA: 'doppler.activation-tensor/v2';
export declare const PARTITION_COMPARISON_SCHEMA: 'doppler.partition-comparison-contract/v1';
export declare const DEFAULT_NUMERICAL_TOLERANCE: number;
export declare const DEFAULT_COSINE_SIMILARITY_MIN: number;
export type ActivationDtype = 'f32' | 'f16';
export type ActivationPrecision = ActivationDtype | 'two-f32' | 'i32';
export interface ActivationLayout {
  precision?: ActivationPrecision; components?: number; elementStrideBytes?: number;
}
export declare function resolveActivationTensorLayout(options: ActivationLayout & { dtype: ActivationDtype | 'i32' }): Required<ActivationLayout>;
export interface PartitionTensorContract extends ActivationLayout {
  type: 'token-ids' | 'activation-tensor' | 'logits';
  rank: number;
  shapeDescription: string;
  dtype: ActivationDtype | 'i32';
  hiddenSize?: number;
  vocabSize?: number;
}
export interface LayerPartition {
  index: number;
  layerRange: readonly number[];
  layerCount: number;
  hasEmbedding: boolean;
  hasLmHead: boolean;
  inputContract: PartitionTensorContract;
  outputContract: PartitionTensorContract;
}
export interface LayerPartitionPlan {
  readonly schema: typeof LAYER_PARTITION_SCHEMA | typeof PAIRED_LAYER_PARTITION_SCHEMA;
  readonly modelId: string;
  /** Exact prepared adapter, bound into the plan digest; absent means base-only. */
  readonly adapterIdentities?: readonly string[];
  readonly totalLayers: number;
  readonly hiddenSize: number;
  readonly vocabSize: number;
  readonly splitLayer: number;
  readonly activationDtype: ActivationDtype;
  readonly activationPrecision?: ActivationPrecision;
  readonly activationComponents?: number;
  readonly activationElementStrideBytes?: number;
  readonly partitions: readonly Readonly<LayerPartition>[];
}
export declare function createLayerPartitionPlan(options: {
  modelId: string; numLayers: number; hiddenSize: number; vocabSize: number;
  adapterIdentities?: readonly string[];
  splitLayer?: number | null; activationDtype?: ActivationDtype; activationPrecision?: ActivationPrecision;
}): LayerPartitionPlan;
/** Explicit ordered layer cuts. This metadata contract does not itself enable resident execution. */
export interface OrderedLayerPartitionPlan extends Omit<LayerPartitionPlan, 'schema' | 'splitLayer'> {
  readonly schema: typeof ORDERED_LAYER_PARTITION_SCHEMA;
  readonly stageCount: number;
  readonly splitLayers: readonly number[];
  readonly activationPrecision: ActivationPrecision;
  readonly activationComponents: number;
  readonly activationElementStrideBytes: number;
}
export type AnyLayerPartitionPlan = LayerPartitionPlan | OrderedLayerPartitionPlan;
export declare function createOrderedLayerPartitionPlan(options: {
  modelId: string; numLayers: number; hiddenSize: number; vocabSize: number;
  adapterIdentities?: readonly string[];
  splitLayers: readonly number[]; activationDtype?: ActivationDtype; activationPrecision?: ActivationPrecision;
}): OrderedLayerPartitionPlan;
export declare function validateOrderedLayerPartitionPlan(manifest: {
  modelId?: unknown; architecture?: unknown;
}, plan: OrderedLayerPartitionPlan): OrderedLayerPartitionPlan;
/** The digest expected by signed resident allocations and TargetPlan qualification. */
export declare function hashLayerPartitionPlan(plan: LayerPartitionPlan | OrderedLayerPartitionPlan): `sha256:${string}`;
export declare function resolveLayerPartition(manifest: {
  modelId?: unknown; architecture?: unknown;
}, allocation: { plan: AnyLayerPartitionPlan; index: number } | null | undefined): Readonly<LayerPartition> | null;
export declare function getPartitionActivationContract(plan: LayerPartitionPlan | OrderedLayerPartitionPlan): Readonly<Required<ActivationLayout> & {
  schema: typeof ACTIVATION_TENSOR_SCHEMA | typeof PAIRED_ACTIVATION_TENSOR_SCHEMA;
  dtype: ActivationDtype; hiddenSize: number;
}>;
