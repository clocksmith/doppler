/**
 * Runtime Config Registry
 *
 * Stores the active RuntimeConfigSchema for the current session.
 * Call setRuntimeConfig() early (before pipeline/loader init) to apply overrides.
 *
 * @module config/runtime
 */

import { createDopplerConfig, DEFAULT_BATCHING_DEFAULTS } from './schema/index.js';
import { log } from '../debug/index.js';

/** @type {import('./schema/index.js').RuntimeConfigSchema} */
let runtimeConfig = createDopplerConfig().runtime;

/**
 * Get the active runtime config (merged with defaults).
 * @returns {import('./schema/index.js').RuntimeConfigSchema}
 */
export function getRuntimeConfig() {
  return runtimeConfig;
}

/**
 * Set the active runtime config.
 * Accepts partial overrides and merges with defaults.
 * @param {Partial<import('./schema/index.js').RuntimeConfigSchema> | import('./schema/index.js').RuntimeConfigSchema} [overrides]
 * @returns {import('./schema/index.js').RuntimeConfigSchema}
 */
export function setRuntimeConfig(overrides) {
  if (!overrides) {
    runtimeConfig = createDopplerConfig().runtime;
    return runtimeConfig;
  }

  const merged = createDopplerConfig({ runtime: overrides }).runtime;

  // Migrate deprecated sampling.maxTokens to batching.maxTokens
  const sampling = /** @type {typeof merged.inference.sampling & { maxTokens?: number }} */ (merged.inference.sampling);
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
 * @returns {import('./schema/index.js').RuntimeConfigSchema}
 */
export function resetRuntimeConfig() {
  runtimeConfig = createDopplerConfig().runtime;
  return runtimeConfig;
}
