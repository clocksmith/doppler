import type { KernelPathSchema } from '../../../config/schema/index.js';
import type { RuntimeConfigSchema } from '../../../config/schema/index.js';
import type { KernelPathSource } from '../../../config/kernel-path-loader.js';

export const PRIMARY_EXECUTION_PLAN_ID: 'primary';
export const FINITENESS_FALLBACK_EXECUTION_PLAN_ID: 'finiteness_fallback';

export interface StaticExecutionPlan {
  id: 'primary' | 'finiteness_fallback';
  source: string;
  kernelPath: KernelPathSchema | null;
  kernelPathId: string | null;
  kernelPathSource: KernelPathSource | 'rule' | 'self' | 'none';
  activationDtype: 'f16' | 'f32';
  finitenessGuardEnabled: boolean;
  finitenessAbsThreshold: number;
  finitenessIncludeNonFinite: boolean;
  deferredRoundingWindowTokens: number;
  defaultDisableCommandBatching: boolean;
  defaultDisableMultiTokenDecode: boolean;
  defaultBatchSize: number;
  defaultStopCheckMode: 'batch' | 'per-token';
  defaultMaxTokens: number;
  readbackInterval: number | null;
  ringTokens: number | null;
  ringStop: number | null;
  ringStaging: number | null;
}

export interface ExecutionPlanState {
  /**
   * Mutable per-generator state; do not share one instance across concurrent generation sessions.
   */
  primaryPlan: StaticExecutionPlan;
  fallbackPlan: StaticExecutionPlan | null;
  activePlanId: 'primary' | 'finiteness_fallback';
}

export interface ExecutionSessionPlan {
  planId: 'primary' | 'finiteness_fallback';
  source: string;
  kernelPath: KernelPathSchema | null;
  kernelPathId: string | null;
  activationDtype: 'f16' | 'f32';
  finitenessGuardEnabled: boolean;
  finitenessAbsThreshold: number;
  finitenessIncludeNonFinite: boolean;
  deferredRoundingWindowTokens: number;
  disableCommandBatching: boolean;
  disableMultiTokenDecode: boolean;
  batchSize: number;
  stopCheckMode: 'batch' | 'per-token';
  maxTokens: number;
  readbackInterval: number | null;
  ringTokens: number | null;
  ringStop: number | null;
  ringStaging: number | null;
  overrides: {
    disableCommandBatching?: boolean;
    disableMultiTokenDecode?: boolean;
    batchSize?: number;
    stopCheckMode?: 'batch' | 'per-token';
    maxTokens?: number;
  };
}

export interface ExecutionPlanCompileOptions {
  runtimeConfig: RuntimeConfigSchema;
  resolvedKernelPath: KernelPathSchema | null;
  kernelPathSource?: KernelPathSource | 'none';
}

export declare function compileExecutionPlanState(options: ExecutionPlanCompileOptions): ExecutionPlanState;
export declare function hasFallbackExecutionPlan(container: { executionPlanState?: ExecutionPlanState } | ExecutionPlanState): boolean;
export declare function resolveActiveExecutionPlan(container: { executionPlanState?: ExecutionPlanState } | ExecutionPlanState): StaticExecutionPlan;
export declare function setActiveExecutionPlan(
  container: { executionPlanState?: ExecutionPlanState } | ExecutionPlanState,
  planId: 'primary' | 'finiteness_fallback'
): StaticExecutionPlan;
export declare function resetActiveExecutionPlan(container: { executionPlanState?: ExecutionPlanState } | ExecutionPlanState): StaticExecutionPlan;
export declare function activateFallbackExecutionPlan(
  container: { executionPlanState?: ExecutionPlanState } | ExecutionPlanState
): StaticExecutionPlan | null;

export declare function resolveExecutionSessionPlan(
  container: { executionPlanState?: ExecutionPlanState } | ExecutionPlanState,
  options?: {
    disableCommandBatching?: boolean;
    disableMultiTokenDecode?: boolean;
    batchSize?: number;
    stopCheckMode?: 'batch' | 'per-token';
    maxTokens?: number;
  }
): ExecutionSessionPlan;

export declare function rebaseExecutionSessionPlan(
  container: { executionPlanState?: ExecutionPlanState } | ExecutionPlanState,
  sessionPlan?: ExecutionSessionPlan | null
): ExecutionSessionPlan;

export declare function isBatchDecodeEnabled(config: {
  batchSize: number;
  useGPU: boolean;
  gpuSamplingAvailable: boolean;
  disableMultiTokenDecode: boolean;
  disableCommandBatching: boolean;
  isBdpaPagedLayout?: boolean;
  finitenessFallbackWindowOpen?: boolean;
}): boolean;

export declare function isDecodeRecorderEnabled(config: {
  hasDevice: boolean;
  debug: boolean;
  disableCommandBatching: boolean;
  kvLayout?: string | null;
}): boolean;
