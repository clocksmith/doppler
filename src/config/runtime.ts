/**
 * Runtime Config Registry
 *
 * Stores the active RuntimeConfigSchema for the current session.
 * Call setRuntimeConfig() early (before pipeline/loader init) to apply overrides.
 *
 * @module config/runtime
 */

import type { RuntimeConfigSchema } from './schema/index.js';
import { createDopplerConfig, DEFAULT_BATCHING_DEFAULTS } from './schema/index.js';

let runtimeConfig: RuntimeConfigSchema = createDopplerConfig().runtime;

/**
 * Get the active runtime config (merged with defaults).
 */
export function getRuntimeConfig(): RuntimeConfigSchema {
  return runtimeConfig;
}

/**
 * Set the active runtime config.
 * Accepts partial overrides and merges with defaults.
 */
export function setRuntimeConfig(
  overrides?: Partial<RuntimeConfigSchema> | RuntimeConfigSchema
): RuntimeConfigSchema {
  if (!overrides) {
    runtimeConfig = createDopplerConfig().runtime;
    return runtimeConfig;
  }

  const merged = createDopplerConfig({ runtime: overrides }).runtime;

  // Back-compat: allow inference.sampling.maxTokens to override batching.maxTokens
  // when batching hasn't been explicitly customized.
  const sampling = merged.inference.sampling as typeof merged.inference.sampling & { maxTokens?: number };
  if (
    sampling.maxTokens !== undefined &&
    merged.inference.batching.maxTokens === DEFAULT_BATCHING_DEFAULTS.maxTokens
  ) {
    merged.inference.batching.maxTokens = sampling.maxTokens;
  }

  runtimeConfig = merged;
  return runtimeConfig;
}

/**
 * Reset runtime config to defaults.
 */
export function resetRuntimeConfig(): RuntimeConfigSchema {
  runtimeConfig = createDopplerConfig().runtime;
  return runtimeConfig;
}
