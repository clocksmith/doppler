/**
 * Execution v1 Schema
 *
 * Compact, explicit execution contract for manifest inference.
 * Kernel declarations + tuple-based step sequences.
 *
 * Design:
 * - Kernels declared once, referenced by key in steps
 * - Steps are tuples [op, kernelKey, weights?]
 * - Phase is structural (decode/prefill arrays)
 * - Layer targeting via group blocks
 * - Execution graph is the only dispatch contract
 *
 * @module config/schema/execution-v1
 */

import type { KVCacheConfigSchema } from './kvcache.schema.js';

// === Primitives ===

export type ExecutionV1Dtype = 'f16' | 'f32';

export interface ExecutionV1PrecisionSchema {
  activationDtype?: ExecutionV1Dtype;
  kvDtype?: ExecutionV1Dtype;
  inputDtype?: ExecutionV1Dtype;
  outputDtype?: ExecutionV1Dtype;
}

// === Kernel Declarations ===

/** A kernel declaration — defines a shader + entry + pinned digest. */
export interface ExecutionV1KernelSchema {
  /** WGSL shader file (e.g., "matmul_gemv_subgroup.wgsl") */
  kernel: string;
  /** Shader entry point (e.g., "main_vec4") */
  entry: string;
  /** SHA-256 digest of normalized shader source + entry */
  digest: string;
  /** Pipeline override constants (optional, baked at pipeline creation) */
  constants?: Record<string, number | boolean>;
  /** Explicit per-step precision contract carried into inline kernel paths. */
  precision?: ExecutionV1PrecisionSchema;
}

/** Map of kernel key → kernel declaration. */
export type ExecutionV1KernelMap = Record<string, ExecutionV1KernelSchema>;

// === Steps ===

/**
 * A step tuple: [op, kernelKey] or [op, kernelKey, weights].
 *
 * - op: operation name (e.g., "q_proj", "attention", "input_norm")
 * - kernelKey: key into the kernels map
 * - weights: tensor name template (e.g., "layer.{L}.self_attn.q_proj")
 */
export type ExecutionV1StepTuple =
  | [op: string, kernelKey: string]
  | [op: string, kernelKey: string, weights: string];

/**
 * A layer group block — targets steps to specific layer indices.
 * Steps outside a group block run on all layers.
 */
export interface ExecutionV1LayerGroupSchema {
  /** Layer indices this group targets */
  layers: number[];
  /** Steps to run on those layers */
  steps: ExecutionV1StepTuple[];
}

/** A step entry is either a tuple (all layers) or a layer group block. */
export type ExecutionV1StepEntry = ExecutionV1StepTuple | ExecutionV1LayerGroupSchema;

// === Pre/Post Layer ===

/**
 * Pre-layer and post-layer steps run once (not per-layer).
 * Same tuple format: [op, kernelKey, weights?].
 */
export type ExecutionV1BoundaryStep = ExecutionV1StepTuple;

// === Session ===

export interface ExecutionV1ComputeDefaultsSchema {
  activationDtype: ExecutionV1Dtype;
  mathDtype: ExecutionV1Dtype;
  accumDtype: ExecutionV1Dtype;
  outputDtype: ExecutionV1Dtype;
}

export type ReadbackMode = 'sequential' | 'overlapped' | 'auto';

export interface ExecutionV1DecodeLoopSchema {
  batchSize: number;
  stopCheckMode: 'per-token' | 'batch';
  readbackInterval: number | null;
  readbackMode: ReadbackMode;
  /**
   * Submit latency threshold in ms. Only consulted when readbackMode === "auto".
   * When the GPU submit probe roundtrip exceeds this value, auto resolves to
   * "overlapped"; otherwise resolves to "sequential".
   *
   * Set to `null` to disable auto (treat as sequential).
   */
  submitLatencyThresholdMs: number | null;
  ringTokens: number | null;
  ringStop: number | null;
  ringStaging: number | null;
  disableCommandBatching?: boolean;
}

export type PerLayerInputMaterializationMode =
  'auto'
  | 'range_backed'
  | 'cpu_resident'
  | 'gpu_resident'
  | 'gpu_split_tables';

export type PerLayerInputRowCacheMode = 'off' | 'lru';

export type PerLayerInputPrefetchMode = 'off' | 'next_token';

export type PerLayerInputGpuUploadMode = 'per_step_slices' | 'per_batch_slices';
export type PerLayerInputHotCacheMode = 'off' | 'prepared_tokens' | 'tokenizer_scores';

export interface ExecutionV1PerLayerInputsRowCacheSchema {
  mode: PerLayerInputRowCacheMode;
  maxRows: number;
  maxBytes: number;
  decodedDtype: ExecutionV1Dtype;
}

export interface ExecutionV1PerLayerInputsPrefetchSchema {
  mode: PerLayerInputPrefetchMode;
  rowsAhead: number;
}

export interface ExecutionV1PerLayerInputsGpuUploadSchema {
  mode: PerLayerInputGpuUploadMode;
  stagingRows: number;
}

export interface ExecutionV1PerLayerInputsHotCacheSchema {
  mode: PerLayerInputHotCacheMode;
  maxTokens: number;
  maxBytes: number;
  outputDtype: ExecutionV1Dtype;
}

export interface ExecutionV1PerLayerInputsSessionSchema {
  materialization: PerLayerInputMaterializationMode;
  rowCache: ExecutionV1PerLayerInputsRowCacheSchema;
  prefetch: ExecutionV1PerLayerInputsPrefetchSchema;
  gpuUpload: ExecutionV1PerLayerInputsGpuUploadSchema;
  hotCache: ExecutionV1PerLayerInputsHotCacheSchema;
}

export interface ExecutionV1SelfSpeculationSchema {
  mode: 'none' | 'self' | 'draft' | 'medusa';
  tokens: number;
  verify: 'greedy';
  threshold: number | null;
  rollbackOnReject: boolean;
}

export interface ExecutionV1SessionSchema {
  compute: {
    defaults: ExecutionV1ComputeDefaultsSchema;
  };
  kvcache: Partial<KVCacheConfigSchema> | null;
  decodeLoop: ExecutionV1DecodeLoopSchema | null;
  perLayerInputs: ExecutionV1PerLayerInputsSessionSchema;
  speculation: ExecutionV1SelfSpeculationSchema | null;
  /** "sync" (default): wait each prefill chunk. "async": queue without waiting. */
  prefillChunkSubmitMode: 'sync' | 'async';
  /** Opt into flash-attention prefill kernel. Requires head_dim=256, f16 KV, contiguous layout. */
  useFlashPrefillAttention: boolean;
  /** Opt into WideTile Q4_K prefill matmul. Requires f32 activations + Q4_K weights + shader-f16 + M>=TILE_M. */
  useWideTileQ4KPrefill: boolean;
  /** Retain Q4_K packed weights alongside dense buffer (mixed materialization). ~50% extra Q4_K memory. */
  retainQ4KMaterialization: boolean;
}

// === Policies ===

export interface ExecutionV1PoliciesSchema {
  unsupportedPrecision: 'error';
  dtypeTransition: 'require_cast_step';
  unresolvedKernel: 'error';
}

// === Top-Level Execution Graph ===

export interface ExecutionV1GraphSchema {
  /** Kernel declarations — each key is a shorthand used in step tuples */
  kernels: ExecutionV1KernelMap;

  /**
   * Whether compileExecutionV1 should lower the graph into an inline kernelPath.
   * Use false when the execution graph is kept for manifest/session contract
   * ownership but the runtime must stay on the existing manifest-first path.
   *
   * Default: true
   */
  inlineKernelPath?: boolean;

  /** Steps run before the layer loop (embed, etc.) */
  preLayer: ExecutionV1BoundaryStep[];

  /** Decode phase layer steps (M=1 optimized kernels) */
  decode: ExecutionV1StepEntry[];

  /** Prefill phase layer steps (batched kernels) */
  prefill: ExecutionV1StepEntry[];

  /** Steps run after the layer loop (final norm, lm_head, sampling) */
  postLayer: ExecutionV1BoundaryStep[];

  /** Fail-fast policies */
  policies: ExecutionV1PoliciesSchema;
}

// === Manifest-Level Config ===

export interface ExecutionV1ConfigSchema {
  /** Schema discriminator — always "doppler.execution/v1" */
  schema: 'doppler.execution/v1';

  /** Session policy (dtypes, KV cache, decode loop) */
  session: ExecutionV1SessionSchema;

  /** The execution graph */
  execution: ExecutionV1GraphSchema;
}

// === Patch (Runtime Overrides) ===

export interface ExecutionV1PatchSetSchema {
  /** Op name to target (matches step[0]) */
  op: string;
  /** Replace kernel key */
  kernelKey?: string;
  /** Replace weights */
  weights?: string;
  /** Target specific layers only (null = all matching ops) */
  layers?: number[] | null;
}

export interface ExecutionV1PatchRemoveSchema {
  /** Op name to remove */
  op: string;
  /** Target specific layers only (null = all matching ops) */
  layers?: number[] | null;
}

export interface ExecutionV1PatchAddSchema {
  /** Step to insert */
  step: ExecutionV1StepTuple;
  /** Insert before this op */
  insertBefore?: string;
  /** Insert after this op */
  insertAfter?: string;
  /** Target specific layers only */
  layers?: number[] | null;
}

export interface ExecutionV1PatchAddKernelSchema {
  /** Kernel key to add */
  key: string;
  /** Kernel declaration */
  kernel: ExecutionV1KernelSchema;
}

export interface ExecutionV1PatchSchema {
  /** Add new kernel declarations */
  addKernels?: ExecutionV1PatchAddKernelSchema[];
  /** Modify existing steps */
  set?: ExecutionV1PatchSetSchema[];
  /** Remove steps */
  remove?: ExecutionV1PatchRemoveSchema[];
  /** Insert new steps */
  add?: ExecutionV1PatchAddSchema[];
}

// === Expanded Form (Runtime Internal) ===

/** Expanded step — what the runtime actually works with after tuple expansion. */
export interface ExecutionV1ExpandedStepSchema {
  op: string;
  kernel: string;
  entry: string;
  digest: string;
  weights: string | null;
  constants: Record<string, number | boolean> | null;
  layers: 'all' | number[];
  phase: 'decode' | 'prefill' | 'both';
  section: 'preLayer' | 'layer' | 'postLayer';
}

// === Constants ===

export declare const EXECUTION_V1_SCHEMA_ID: string;
export declare const READBACK_MODES: readonly ['sequential', 'overlapped', 'auto'];
export declare const PREFILL_CHUNK_SUBMIT_MODES: readonly ['sync', 'async'];
export declare const PER_LAYER_INPUT_MATERIALIZATION_MODES: readonly ['auto', 'range_backed', 'cpu_resident', 'gpu_resident'];
export declare const PER_LAYER_INPUT_ROW_CACHE_MODES: readonly ['off', 'lru'];
export declare const PER_LAYER_INPUT_PREFETCH_MODES: readonly ['off', 'next_token'];
export declare const PER_LAYER_INPUT_GPU_UPLOAD_MODES: readonly ['per_step_slices', 'per_batch_slices'];
export declare const DEFAULT_EXECUTION_V1_COMPUTE_DEFAULTS: ExecutionV1ComputeDefaultsSchema;
export declare const DEFAULT_EXECUTION_V1_SESSION: ExecutionV1SessionSchema;
export declare const DEFAULT_EXECUTION_V1_POLICIES: ExecutionV1PoliciesSchema;
export declare const DEFAULT_EXECUTION_V1_PATCH: ExecutionV1PatchSchema;

// === Validation ===

export declare function isExecutionV1Digest(value: unknown): boolean;

export interface ExpandExecutionV1Options {
  /**
   * When provided, any op not in this set triggers a warning (or throw in
   * strict mode) with the op name and step index.
   */
  knownOps?: ReadonlySet<string> | null;
  /** When true, unknown ops throw instead of warn. Default false. */
  strict?: boolean;
}

/** Validate and expand a v1 execution graph into runtime-ready expanded steps. */
export declare function expandExecutionV1(
  graph: ExecutionV1GraphSchema,
  options?: ExpandExecutionV1Options
): ExecutionV1ExpandedStepSchema[];

/** Check if a manifest inference object uses execution v1. */
export declare function hasExecutionV1(
  inference: { schema?: string | null }
): boolean;
