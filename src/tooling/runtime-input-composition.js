import { mergeRuntimeValues } from '../config/runtime-merge.js';

export function resolveRuntimeFromConfig(config) {
  if (!config || typeof config !== 'object') return null;
  if (config.runtime && typeof config.runtime === 'object') return config.runtime;
  if (config.shared || config.loading || config.inference || config.emulation) return config;
  return null;
}

function mergeRuntimePatch(runtimeBridge, patch) {
  if (!patch) return;
  const mergedRuntime = mergeRuntimeValues(runtimeBridge.getRuntimeConfig(), patch);
  runtimeBridge.setRuntimeConfig(mergedRuntime);
}

function requireRuntimeBridge(runtimeBridge) {
  if (!runtimeBridge?.setRuntimeConfig) {
    throw new Error('runtime bridge must provide setRuntimeConfig().');
  }
  if (typeof runtimeBridge.getRuntimeConfig !== 'function') {
    throw new Error('runtime bridge must provide getRuntimeConfig().');
  }
}

async function applyConfigChain(configChain, runtimeBridge, loadRuntimeConfigFromRef, options) {
  if (!Array.isArray(configChain) || configChain.length === 0) {
    return;
  }
  if (typeof loadRuntimeConfigFromRef !== 'function') {
    throw new Error('runtime input composition does not support configChain on this surface.');
  }
  for (const ref of configChain) {
    const loaded = await loadRuntimeConfigFromRef(ref, options);
    const runtime = resolveRuntimeFromConfig(loaded);
    if (!runtime) {
      throw new Error(`Loaded runtime config "${ref}" is missing runtime fields.`);
    }
    mergeRuntimePatch(runtimeBridge, runtime);
  }
}

async function applyRuntimePreset(runtimePreset, applyPreset, options) {
  if (!runtimePreset) {
    return;
  }
  if (typeof applyPreset !== 'function') {
    throw new Error('runtime input composition does not support runtimePreset on this surface.');
  }
  await applyPreset(runtimePreset, options);
}

async function applyRuntimeConfigUrl(runtimeConfigUrl, applyConfigFromUrl, options) {
  if (!runtimeConfigUrl) {
    return;
  }
  if (typeof applyConfigFromUrl !== 'function') {
    throw new Error('runtime input composition does not support runtimeConfigUrl on this surface.');
  }
  await applyConfigFromUrl(runtimeConfigUrl, options);
}

export async function applyOrderedRuntimeInputs(runtimeBridge, inputs = {}, handlers = {}, options = {}) {
  requireRuntimeBridge(runtimeBridge);

  await applyConfigChain(
    inputs.configChain,
    runtimeBridge,
    handlers.loadRuntimeConfigFromRef,
    options
  );
  await applyRuntimePreset(inputs.runtimePreset, handlers.applyRuntimePreset, options);
  await applyRuntimeConfigUrl(inputs.runtimeConfigUrl, handlers.applyRuntimeConfigFromUrl, options);

  if (inputs.runtimeConfig) {
    const runtime = resolveRuntimeFromConfig(inputs.runtimeConfig);
    if (!runtime) {
      throw new Error('runtimeConfig is missing runtime fields');
    }
    mergeRuntimePatch(runtimeBridge, runtime);
  }

  const runtimeContractPatch = typeof inputs.runtimeContractPatch === 'function'
    ? inputs.runtimeContractPatch()
    : (inputs.runtimeContractPatch ?? null);
  mergeRuntimePatch(runtimeBridge, runtimeContractPatch);
}
