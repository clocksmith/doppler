import type { RuntimeConfigSchema, KernelPathSchema } from '../../../config/schema/index.js';
import type { Manifest, ParsedModelConfig } from './config.js';
import type { ExecutionPlanState } from './execution-plan.js';
import type { ExecutionV1CompiledState } from './execution-v1.js';

export const RESOLVED_RUNTIME_SESSION_SCHEMA: 'doppler.resolved-runtime-session/v1';

export interface ResolvedRuntimeSession {
  readonly schema: typeof RESOLVED_RUNTIME_SESSION_SCHEMA;
  readonly id: string;
  readonly model: {
    readonly id: string;
    readonly type: string;
    readonly architecture: string;
    readonly numLayers: number;
    readonly hiddenSize: number;
    readonly numHeads: number;
    readonly numKVHeads: number;
    readonly headDim: number;
  };
  readonly manifestInference: Readonly<Record<string, unknown>> | null;
  readonly runtime: {
    readonly session: Readonly<Record<string, unknown>>;
    readonly compute: Readonly<Record<string, unknown>>;
  };
  readonly kernelPath: {
    readonly id: string | null;
    readonly source: string;
    readonly hash: string | null;
    readonly definition: Readonly<Record<string, unknown>> | null;
  };
  readonly capabilityPolicy: Readonly<Record<string, unknown>> | null;
  readonly laneIntegrity: Readonly<Record<string, unknown>> | null;
  readonly execution: Readonly<Record<string, unknown>>;
  readonly dtypes: {
    readonly activation: string;
    readonly output: string;
    readonly kv: string;
    readonly math: string | null;
    readonly accumulation: string | null;
  };
}

export function createResolvedRuntimeSession(options: {
  manifest: Manifest;
  modelConfig: ParsedModelConfig;
  runtimeConfig: RuntimeConfigSchema;
  resolvedKernelPath?: KernelPathSchema | null;
  kernelPathSource?: string;
  executionV1State?: ExecutionV1CompiledState | null;
  executionPlanState: ExecutionPlanState;
}): ResolvedRuntimeSession;

export function resolveAttentionRuntimeSession(state: {
  resolvedRuntimeSession?: ResolvedRuntimeSession | null;
}): Readonly<Record<string, unknown>>;
