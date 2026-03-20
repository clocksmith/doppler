/**
 * Execution v1 Schema
 *
 * Compact, explicit execution contract for manifest inference.
 * Replaces v0 with: kernel declarations + tuple-based step sequences.
 *
 * Key differences from v0:
 * - Kernels declared once, referenced by key in steps
 * - Steps are tuples [op, kernelKey, weights?] not objects
 * - Phase is structural (decode/prefill arrays) not per-step
 * - Layer targeting via group blocks, not per-step field
 * - No defaultKernelPath — execution graph is the only dispatch contract
 *
 * @module config/schema/execution-v1
 */

import type { KVCacheConfigSchema } from './kvcache.schema.js';

// === Primitives ===

export type ExecutionV1Dtype = 'f16' | 'f32';

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

// === Session Defaults ===

export interface ExecutionV1ComputeDefaultsSchema {
  activationDtype: ExecutionV1Dtype;
  mathDtype: ExecutionV1Dtype;
  accumDtype: ExecutionV1Dtype;
  outputDtype: ExecutionV1Dtype;
}

export interface ExecutionV1DecodeLoopSchema {
  batchSize: number;
  stopCheckMode: 'per-token' | 'batch';
  readbackInterval: number | null;
  ringTokens: number | null;
  ringStop: number | null;
  ringStaging: number | null;
  disableCommandBatching?: boolean;
}

export interface ExecutionV1SessionDefaultsSchema {
  compute: {
    defaults: ExecutionV1ComputeDefaultsSchema;
  };
  kvcache: Partial<KVCacheConfigSchema> | null;
  decodeLoop: ExecutionV1DecodeLoopSchema | null;
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

  /** Session defaults (dtypes, KV cache, decode loop) */
  sessionDefaults: ExecutionV1SessionDefaultsSchema;

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
export declare const DEFAULT_EXECUTION_V1_COMPUTE_DEFAULTS: ExecutionV1ComputeDefaultsSchema;
export declare const DEFAULT_EXECUTION_V1_SESSION_DEFAULTS: ExecutionV1SessionDefaultsSchema;
export declare const DEFAULT_EXECUTION_V1_POLICIES: ExecutionV1PoliciesSchema;

// === Validation ===

export declare function isExecutionV1Digest(value: unknown): boolean;

/** Validate and expand a v1 execution graph into runtime-ready expanded steps. */
export declare function expandExecutionV1(
  graph: ExecutionV1GraphSchema
): ExecutionV1ExpandedStepSchema[];

/** Check if a manifest inference object uses execution v1. */
export declare function hasExecutionV1(
  inference: { schema?: string | null }
): boolean;
