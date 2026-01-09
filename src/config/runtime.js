import { createDopplerConfig } from './schema/index.js';

let runtimeConfig = createDopplerConfig().runtime;

export function getRuntimeConfig() {
  return runtimeConfig;
}

export function setRuntimeConfig(overrides) {
  if (!overrides) {
    runtimeConfig = createDopplerConfig().runtime;
    return runtimeConfig;
  }

  assertNoDeprecatedRuntimeKeys(overrides);

  const merged = createDopplerConfig({ runtime: overrides }).runtime;

  runtimeConfig = merged;
  return runtimeConfig;
}

export function resetRuntimeConfig() {
  runtimeConfig = createDopplerConfig().runtime;
  return runtimeConfig;
}

function assertNoDeprecatedRuntimeKeys(overrides) {
  if (!overrides || typeof overrides !== 'object') {
    return;
  }

  if (/** @type {Record<string, unknown>} */ (overrides).debug !== undefined) {
    throw new Error('runtime.debug is removed; use runtime.shared.debug');
  }

  const loading = /** @type {{ debug?: unknown } | undefined} */ (overrides).loading;
  if (loading?.debug !== undefined) {
    throw new Error('runtime.loading.debug is removed; use runtime.shared.debug');
  }

  const inference = /** @type {{ debug?: unknown; sampling?: { maxTokens?: unknown } }} */ (overrides).inference;
  if (inference?.debug !== undefined) {
    throw new Error('runtime.inference.debug is removed; use runtime.shared.debug');
  }
  if (inference?.sampling?.maxTokens !== undefined) {
    throw new Error('sampling.maxTokens is removed; use inference.batching.maxTokens');
  }
}
