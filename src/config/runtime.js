import { ERROR_CODES, createDopplerError } from '../errors/doppler-error.js';
import { createDopplerConfig, setKernelThresholds } from './schema/index.js';
import { validateRuntimeConfig, validateRuntimeOverrides } from './param-validator.js';
import { isPlainObject } from '../formats/plain-object.js';

let runtimeConfig = createDopplerConfig().runtime;
const runtimeScopes = [];
setKernelThresholds(runtimeConfig.shared.kernelThresholds);

export function getRuntimeConfig() {
  return runtimeScopes.at(-1)?.config ?? runtimeConfig;
}

export function setRuntimeConfig(overrides) {
  runtimeConfig = resolveRuntimeConfig(overrides);
  setKernelThresholds(getRuntimeConfig().shared.kernelThresholds);
  return runtimeConfig;
}

// Defaults affect future sessions; compatibility execution leases its snapshot.
export function enterRuntimeConfig(config) {
  validateRuntimeConfig(config);
  const entry = { config, active: true };
  runtimeScopes.push(entry);
  setKernelThresholds(config.shared.kernelThresholds);
  return () => {
    entry.active = false;
    while (runtimeScopes.length && !runtimeScopes.at(-1).active) runtimeScopes.pop();
    setKernelThresholds(getRuntimeConfig().shared.kernelThresholds);
  };
}

// Pure resolution is also used by session owners; it never installs defaults.
export function resolveRuntimeConfig(overrides) {
  if (overrides === undefined || overrides === null) {
    return createDopplerConfig().runtime;
  }

  if (!isPlainObject(overrides)) {
    throw createDopplerError(ERROR_CODES.CONFIG_INVALID, 'DopplerConfigError: runtime overrides must be an object when provided.');
  }

  try {
    assertNoDeprecatedRuntimeKeys(overrides);
    validateRuntimeOverrides(overrides);

    const merged = createDopplerConfig({ runtime: overrides }).runtime;

    validateRuntimeConfig(merged);
    return merged;
  } catch (error) {
    if (error?.code || !(error instanceof Error) || !error.message.startsWith('DopplerConfigError:')) throw error;
    throw createDopplerError(ERROR_CODES.CONFIG_INVALID, error.message, { cause: error });
  }
}

export function snapshotRuntimeConfig(config) {
  function freeze(value) {
    if (!Array.isArray(value) && !isPlainObject(value)) return value;
    for (const child of Object.values(value)) freeze(child);
    return Object.freeze(value);
  }
  return freeze(resolveRuntimeConfig(config));
}

export function resetRuntimeConfig() {
  return setRuntimeConfig();
}

function assertNoDeprecatedRuntimeKeys(overrides) {
  if (!overrides || typeof overrides !== 'object') {
    return;
  }

  // Deprecated in v0.8 — debug config consolidated under runtime.shared.debug
  if (overrides.debug !== undefined) {
    throw createDopplerError(ERROR_CODES.CONFIG_INVALID, 'runtime.debug is removed; use runtime.shared.debug');
  }

  const loading = overrides.loading;
  // Deprecated in v0.8 — debug config consolidated under runtime.shared.debug
  if (loading?.debug !== undefined) {
    throw createDopplerError(ERROR_CODES.CONFIG_INVALID, 'runtime.loading.debug is removed; use runtime.shared.debug');
  }

  const inference = overrides.inference;
  // Deprecated in v0.8 — debug config consolidated under runtime.shared.debug
  if (inference?.debug !== undefined) {
    throw createDopplerError(ERROR_CODES.CONFIG_INVALID, 'runtime.inference.debug is removed; use runtime.shared.debug');
  }
  // Deprecated in v0.9 — sampling.maxTokens replaced by generation.maxTokens
  if (inference?.sampling?.maxTokens !== undefined) {
    throw createDopplerError(ERROR_CODES.CONFIG_INVALID, 'sampling.maxTokens is removed; use inference.generation.maxTokens');
  }
  // Deprecated in v0.9 — session.maxNewTokens replaced by inference.generation.maxTokens
  if (inference?.session?.maxNewTokens !== undefined) {
    throw createDopplerError(ERROR_CODES.CONFIG_INVALID, 'inference.session.maxNewTokens is not a supported runtime config key; use inference.generation.maxTokens');
  }
  // Deprecated in v0.9 — command batching policy now lives in the session decode loop
  if (inference?.generation?.disableCommandBatching !== undefined) {
    throw createDopplerError(ERROR_CODES.CONFIG_INVALID,
      'inference.generation.disableCommandBatching is removed; ' +
      'use inference.session.decodeLoop.disableCommandBatching'
    );
  }
}
