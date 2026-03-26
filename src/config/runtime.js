import { createDopplerConfig, setKernelThresholds } from './schema/index.js';
import { validateRuntimeConfig, validateRuntimeOverrides } from './param-validator.js';
import { isPlainObject } from '../utils/plain-object.js';

let runtimeConfig = createDopplerConfig().runtime;
setKernelThresholds(runtimeConfig.shared.kernelThresholds);

export function getRuntimeConfig() {
  return runtimeConfig;
}

export function setRuntimeConfig(overrides) {
  if (overrides === undefined || overrides === null) {
    runtimeConfig = createDopplerConfig().runtime;
    setKernelThresholds(runtimeConfig.shared.kernelThresholds);
    return runtimeConfig;
  }

  if (!isPlainObject(overrides)) {
    throw new Error('DopplerConfigError: runtime overrides must be an object when provided.');
  }

  assertNoDeprecatedRuntimeKeys(overrides);
  validateRuntimeOverrides(overrides);

  const merged = createDopplerConfig({ runtime: overrides }).runtime;

  validateRuntimeConfig(merged);
  runtimeConfig = merged;
  setKernelThresholds(runtimeConfig.shared.kernelThresholds);
  return runtimeConfig;
}

export function resetRuntimeConfig() {
  runtimeConfig = createDopplerConfig().runtime;
  setKernelThresholds(runtimeConfig.shared.kernelThresholds);
  return runtimeConfig;
}

function assertNoDeprecatedRuntimeKeys(overrides) {
  if (!overrides || typeof overrides !== 'object') {
    return;
  }

  // Deprecated in v0.8 — debug config consolidated under runtime.shared.debug
  if (overrides.debug !== undefined) {
    throw new Error('runtime.debug is removed; use runtime.shared.debug');
  }

  const loading = overrides.loading;
  // Deprecated in v0.8 — debug config consolidated under runtime.shared.debug
  if (loading?.debug !== undefined) {
    throw new Error('runtime.loading.debug is removed; use runtime.shared.debug');
  }

  const inference = overrides.inference;
  // Deprecated in v0.8 — debug config consolidated under runtime.shared.debug
  if (inference?.debug !== undefined) {
    throw new Error('runtime.inference.debug is removed; use runtime.shared.debug');
  }
  // Deprecated in v0.9 — sampling.maxTokens replaced by generation.maxTokens
  if (inference?.sampling?.maxTokens !== undefined) {
    throw new Error('sampling.maxTokens is removed; use inference.generation.maxTokens');
  }
  // Deprecated in v0.9 — session.maxNewTokens replaced by inference.generation.maxTokens
  if (inference?.session?.maxNewTokens !== undefined) {
    throw new Error('inference.session.maxNewTokens is not a supported runtime config key; use inference.generation.maxTokens');
  }
}
