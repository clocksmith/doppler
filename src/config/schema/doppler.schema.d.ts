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

import type { ResolvedConfigSchema } from './preset.schema.js';
import type { LoadingConfigSchema } from './loading.schema.js';
import type { InferenceDefaultsConfigSchema } from './inference-defaults.schema.js';
import type { SharedRuntimeConfigSchema } from './shared-runtime.schema.js';

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
  inference: InferenceDefaultsConfigSchema;
}

/** Default runtime configuration */
export declare const DEFAULT_RUNTIME_CONFIG: RuntimeConfigSchema;

/**
 * Master Doppler configuration schema.
 *
 * Combines model-specific configuration (resolved from preset + manifest)
 * with runtime configuration (engine settings) and platform overrides.
 */
export interface DopplerConfigSchema {
  /** Model-specific configuration (from preset + manifest) */
  model?: ResolvedConfigSchema;

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
