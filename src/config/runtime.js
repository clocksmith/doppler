import { createDopplerConfig, setKernelThresholds } from './schema/index.js';
import { validateRuntimeOverrides } from './param-validator.js';

let runtimeConfig = createDopplerConfig().runtime;
setKernelThresholds(runtimeConfig.shared.kernelThresholds);

export function getRuntimeConfig() {
  return runtimeConfig;
}

export function setRuntimeConfig(overrides) {
  if (!overrides) {
    runtimeConfig = createDopplerConfig().runtime;
    setKernelThresholds(runtimeConfig.shared.kernelThresholds);
    return runtimeConfig;
  }

  assertNoDeprecatedRuntimeKeys(overrides);
  validateRuntimeOverrides(overrides);

  const merged = createDopplerConfig({ runtime: overrides }).runtime;

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

  if (overrides.debug !== undefined) {
    throw new Error('runtime.debug is removed; use runtime.shared.debug');
  }

  const loading = overrides.loading;
  if (loading?.debug !== undefined) {
    throw new Error('runtime.loading.debug is removed; use runtime.shared.debug');
  }

  const inference = overrides.inference;
  if (inference?.debug !== undefined) {
    throw new Error('runtime.inference.debug is removed; use runtime.shared.debug');
  }
  if (inference?.sampling?.maxTokens !== undefined) {
    throw new Error('sampling.maxTokens is removed; use inference.batching.maxTokens');
  }
}
