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
import { log } from '../debug/index.js';

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

  // Migrate deprecated sampling.maxTokens to batching.maxTokens
  const sampling = merged.inference.sampling as typeof merged.inference.sampling & { maxTokens?: number };
  if (sampling.maxTokens !== undefined) {
    log.warn('Config', 'inference.sampling.maxTokens is deprecated, use inference.batching.maxTokens instead');
    // Only migrate if batching.maxTokens is still at default (user didn't explicitly set it)
    if (merged.inference.batching.maxTokens === DEFAULT_BATCHING_DEFAULTS.maxTokens) {
      merged.inference.batching.maxTokens = sampling.maxTokens;
      log.debug('Config', `Migrated sampling.maxTokens=${sampling.maxTokens} to batching.maxTokens`);
    }
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
