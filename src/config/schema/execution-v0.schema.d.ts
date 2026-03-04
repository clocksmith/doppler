/**
 * Execution v0 Schema
 *
 * Manifest/runtime contract for explicit per-step execution and precision policy.
 *
 * @module config/schema/execution-v0
 */

import type { KVCacheConfigSchema } from './kvcache.schema.js';

export type ExecutionV0Dtype = 'f16' | 'f32';
export type ExecutionV0Phase = 'prefill' | 'decode' | 'both';
export type ExecutionV0Section = 'preLayer' | 'layer' | 'postLayer' | 'sampling';
export type ExecutionV0LayerSelector = 'all' | number[];

export interface ExecutionV0KernelRefSchema {
  id: string;
  version: string;
  digest: string;
}

export interface ExecutionV0PrecisionSchema {
  inputDtype?: ExecutionV0Dtype;
  mathDtype?: ExecutionV0Dtype;
  accumDtype?: ExecutionV0Dtype;
  outputDtype?: ExecutionV0Dtype;
}

export interface ExecutionV0KVIO {
  readDtype: ExecutionV0Dtype;
  writeDtype: ExecutionV0Dtype;
}

export interface ExecutionV0KernelProfileSchema {
  kernelRef: ExecutionV0KernelRefSchema;
  precision?: ExecutionV0PrecisionSchema;
  kvIO?: ExecutionV0KVIO;
}

export interface ExecutionV0ComputeDefaultsSchema extends ExecutionV0PrecisionSchema {
  activationDtype: ExecutionV0Dtype;
  mathDtype: ExecutionV0Dtype;
  accumDtype: ExecutionV0Dtype;
  outputDtype: ExecutionV0Dtype;
}

export interface ExecutionV0ComputeSessionSchema {
  defaults: ExecutionV0ComputeDefaultsSchema;
  kernelProfiles: ExecutionV0KernelProfileSchema[];
}

export interface ExecutionV0DecodeLoopSchema {
  batchSize: number;
  stopCheckMode: 'per-token' | 'batch';
  readbackInterval: number | null;
  ringTokens: number | null;
  ringStop: number | null;
  ringStaging: number | null;
  disableCommandBatching?: boolean;
}

export interface ExecutionV0SessionDefaultsSchema {
  compute: ExecutionV0ComputeSessionSchema;
  kvcache: Partial<KVCacheConfigSchema> | null;
  decodeLoop: ExecutionV0DecodeLoopSchema | null;
}

export type ExecutionV0PolicyPrecedence = 'step_then_kernel_profile_then_session_default';
export type ExecutionV0PolicyUnsupported = 'error';
export type ExecutionV0PolicyTransition = 'require_cast_step';
export type ExecutionV0PolicyUnresolvedKernel = 'error';

export interface ExecutionV0PoliciesSchema {
  precisionPrecedence: ExecutionV0PolicyPrecedence;
  unsupportedPrecision: ExecutionV0PolicyUnsupported;
  dtypeTransition: ExecutionV0PolicyTransition;
  unresolvedKernel: ExecutionV0PolicyUnresolvedKernel;
}

export interface ExecutionV0StepBaseSchema {
  id: string;
  phase: ExecutionV0Phase;
  section: ExecutionV0Section;
  src: string;
  dst: string;
  layers: ExecutionV0LayerSelector;
}

export interface ExecutionV0KernelStepSchema extends ExecutionV0StepBaseSchema {
  op: string;
  kernelRef: ExecutionV0KernelRefSchema;
  kernel: string;
  entry?: string;
  weights?: string;
  constants?: Record<string, number | boolean>;
  precision?: ExecutionV0PrecisionSchema;
  kvIO?: ExecutionV0KVIO;
}

export interface ExecutionV0CastStepSchema extends ExecutionV0StepBaseSchema {
  op: 'cast';
  kernelRef?: never;
  kernel?: never;
  entry?: never;
  weights?: never;
  constants?: never;
  precision?: ExecutionV0PrecisionSchema;
  fromDtype?: ExecutionV0Dtype;
  toDtype: ExecutionV0Dtype;
}

export type ExecutionV0StepSchema = ExecutionV0KernelStepSchema | ExecutionV0CastStepSchema;

export interface ExecutionV0GraphSchema {
  steps: ExecutionV0StepSchema[];
  policies: ExecutionV0PoliciesSchema;
}

export interface ExecutionV0ModelSchema {
  attention: {
    causal?: boolean;
    slidingWindow?: number | null;
    queryKeyNorm?: boolean;
    attentionOutputGate?: boolean;
    attnLogitSoftcapping?: number | null;
    queryPreAttnScalar?: number;
  };
}

export interface ExecutionV0ConfigSchema {
  model: ExecutionV0ModelSchema | null;
  sessionDefaults: ExecutionV0SessionDefaultsSchema;
  execution: ExecutionV0GraphSchema;
}

export interface ExecutionV0PatchSetSchema {
  id: string;
  precision?: ExecutionV0PrecisionSchema;
  kvIO?: ExecutionV0KVIO;
  constants?: Record<string, number | boolean>;
  entry?: string;
}

export interface ExecutionV0PatchRemoveSchema {
  id: string;
}

export interface ExecutionV0PatchAddSchema {
  step: ExecutionV0StepSchema;
  insertBefore?: string;
  insertAfter?: string;
}

export interface ExecutionV0PatchSchema {
  set: ExecutionV0PatchSetSchema[];
  remove: ExecutionV0PatchRemoveSchema[];
  add: ExecutionV0PatchAddSchema[];
}

export declare const EXECUTION_V0_SCHEMA_ID: string;
export declare const EXECUTION_V0_HASH_PATTERN: RegExp;
export declare const EXECUTION_V0_SEMVER_PATTERN: RegExp;

export declare const DEFAULT_EXECUTION_V0_COMPUTE_DEFAULTS: ExecutionV0ComputeDefaultsSchema;
export declare const DEFAULT_EXECUTION_V0_SESSION_DEFAULTS: ExecutionV0SessionDefaultsSchema;
export declare const DEFAULT_EXECUTION_V0_POLICIES: ExecutionV0PoliciesSchema;
export declare const DEFAULT_EXECUTION_V0_CONFIG: ExecutionV0ConfigSchema;
export declare const DEFAULT_EXECUTION_V0_PATCH: ExecutionV0PatchSchema;

export declare function isExecutionV0Digest(value: unknown): boolean;
export declare function isExecutionV0Semver(value: unknown): boolean;

export interface ResolvedExecutionV0StepSchema extends ExecutionV0StepSchema {
  precision: Required<ExecutionV0PrecisionSchema>;
  kvIO: ExecutionV0KVIO | null;
}

export interface ExecutionV0ResolveSourceSchema {
  source: 'manifest' | 'runtime.session' | 'runtime.patch' | 'kernelProfile' | 'derived';
}

export type ExecutionV0FieldSourceMap = Record<string, ExecutionV0ResolveSourceSchema>;

export interface ExecutionV0ResolvedSourcesSchema {
  session: ExecutionV0FieldSourceMap;
  steps: Record<string, ExecutionV0FieldSourceMap>;
}
