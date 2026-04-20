/**
 * Doppler Config Schema
 *
 * Master configuration schema that composes all runtime configs together.
 * This provides a single unified interface for configuring the entire
 * Doppler inference engine.
 *
 * Individual configs remain importable for subsystems that only need
 * their specific domain. This master config is for:
 * - Serializing/restoring full engine state
 * - Configuration management UIs
 * - Debugging/logging full config state
 *
 * @module config/schema/doppler
 */

import type { LoadingConfigSchema } from './loading.schema.js';
import type { ExecutionV1PatchSchema, ExecutionV1SessionSchema } from './execution-v1.schema.js';
import type { SharedRuntimeConfigSchema } from './shared-runtime.schema.js';
import type { EmulationConfigSchema } from './emulation.schema.js';

export interface LargeWeightConfigSchema {
  enabled: boolean;
  safetyRatio: number;
  preferF16: boolean;
  lmHeadChunkRows: number | null;
  gpuResidentOverrides: string[] | null;
}

export interface RuntimeInferenceConfigSchema {
  prompt?: string | null;
  debugTokens?: boolean;
  batching: Record<string, unknown>;
  sampling: Record<string, unknown>;
  compute: Record<string, unknown>;
  tokenizer: Record<string, unknown>;
  largeWeights: LargeWeightConfigSchema;
  kvcache: Record<string, unknown>;
  diffusion: Record<string, unknown>;
  energy: Record<string, unknown>;
  moe: Record<string, unknown>;
  speculative: Record<string, unknown>;
  generation: Record<string, unknown>;
  chatTemplate: Record<string, unknown>;
  session: Partial<ExecutionV1SessionSchema>;
  executionPatch: ExecutionV1PatchSchema | Record<string, unknown>;
  kernelPath?: unknown;
  kernelPathSource?: 'runtime' | 'config' | 'model' | 'manifest' | 'none';
  kernelPathPolicy?: Record<string, unknown> | null;
  modelOverrides?: Record<string, unknown>;
  pipeline?: Record<string, unknown> | null;
}

/**
 * Runtime configuration schema.
 *
 * Contains all configurable settings that are independent of the model.
 * These settings control engine behavior regardless of which model is loaded.
 */
export interface RuntimeConfigSchema {
  /** Cross-cutting runtime settings shared by loader + inference */
  shared: SharedRuntimeConfigSchema;

  /** OPFS paths, shard cache, memory management */
  loading: LoadingConfigSchema;

  /** Batching, sampling, tokenizer defaults */
  inference: RuntimeInferenceConfigSchema;

  /** NVIDIA superchip emulation settings */
  emulation: EmulationConfigSchema;
}

/** Chat-template defaults (enabled/type/thinking tri-state) */
export interface ChatTemplateDefaultConfigSchema {
  enabled: boolean | null;
  type: string | null;
  thinking: boolean;
}

/** Default runtime configuration */
export declare const DEFAULT_LARGE_WEIGHT_CONFIG: LargeWeightConfigSchema;
export declare const DEFAULT_CHAT_TEMPLATE_CONFIG: ChatTemplateDefaultConfigSchema;
export declare const DEFAULT_RUNTIME_CONFIG: RuntimeConfigSchema;

/**
 * Master Doppler configuration schema.
 *
 * Combines model-specific configuration resolved from manifest + runtime config
 * with runtime configuration (engine settings) and platform overrides.
 */
export interface DopplerConfigSchema {
  /** Model-specific configuration (reserved, no longer family-registry driven) */
  model?: Record<string, unknown>;

  /** Runtime configuration (engine settings) */
  runtime: RuntimeConfigSchema;
}

export interface DopplerConfigOverrides extends Partial<Omit<DopplerConfigSchema, 'runtime'>> {
  runtime?: Partial<RuntimeConfigSchema>;
}

/** Default Doppler configuration (no model loaded) */
export declare const DEFAULT_DOPPLER_CONFIG: DopplerConfigSchema;

/**
 * Create a Doppler configuration with optional overrides.
 *
 * Merges provided overrides with defaults, performing a deep merge
 * on nested objects.
 */
export declare function createDopplerConfig(
  overrides?: DopplerConfigOverrides
): DopplerConfigSchema;
