/**
 * Runtime Config Registry
 *
 * Stores the active RuntimeConfigSchema for the current session.
 * Call setRuntimeConfig() early (before pipeline/loader init) to apply overrides.
 *
 * @module config/runtime
 */

import type { RuntimeConfigSchema } from './schema/index.js';

/**
 * Get the active runtime config (merged with defaults).
 */
export function getRuntimeConfig(): RuntimeConfigSchema;
export function enterRuntimeConfig(config: RuntimeConfigSchema): () => void;

/**
 * Set the active runtime config.
 * Accepts partial overrides and merges with defaults.
 */
export function setRuntimeConfig(
  overrides?: Partial<RuntimeConfigSchema> | RuntimeConfigSchema
): RuntimeConfigSchema;

/**
 * Reset runtime config to defaults.
 */
export function resetRuntimeConfig(): RuntimeConfigSchema;

export function resolveRuntimeConfig(overrides?: Partial<RuntimeConfigSchema> | null): RuntimeConfigSchema;
export function snapshotRuntimeConfig(config: RuntimeConfigSchema): Readonly<RuntimeConfigSchema>;
