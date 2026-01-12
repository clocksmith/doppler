import { buildRuntimeContractPatch } from './command-api.js';
import { mergeRuntimeValues } from '../config/runtime-merge.js';

function cloneRuntimeConfig(runtimeConfig) {
  if (runtimeConfig == null) return runtimeConfig;
  if (typeof structuredClone === 'function') {
    return structuredClone(runtimeConfig);
  }
  return JSON.parse(JSON.stringify(runtimeConfig));
}

function resetRuntimeState(runtimeBridge) {
  if (!runtimeBridge?.setRuntimeConfig) {
    throw new Error('runtime bridge must provide setRuntimeConfig().');
  }

  if (typeof runtimeBridge.resetRuntimeConfig === 'function') {
    runtimeBridge.resetRuntimeConfig();
    return;
  }

  runtimeBridge.setRuntimeConfig(null);
}

function mergeRuntimePatch(runtimeBridge, patch) {
  if (!patch) return;
  const mergedRuntime = mergeRuntimeValues(runtimeBridge.getRuntimeConfig(), patch);
  runtimeBridge.setRuntimeConfig(mergedRuntime);
}

function snapshotRuntimeState(runtimeBridge) {
  return {
    runtimeConfig: cloneRuntimeConfig(runtimeBridge.getRuntimeConfig()),
    activeKernelPath: runtimeBridge.getActiveKernelPath
      ? runtimeBridge.getActiveKernelPath()
      : null,
    activeKernelPathSource: runtimeBridge.getActiveKernelPathSource
      ? runtimeBridge.getActiveKernelPathSource()
      : 'none',
  };
}

function restoreRuntimeState(runtimeBridge, snapshot) {
  if (!snapshot) {
    return;
  }

  if (snapshot.runtimeConfig) {
    runtimeBridge.setRuntimeConfig(snapshot.runtimeConfig);
  } else {
    runtimeBridge.resetRuntimeConfig();
  }

  if (
    snapshot.activeKernelPath !== null
    && typeof runtimeBridge.setActiveKernelPath === 'function'
  ) {
    runtimeBridge.setActiveKernelPath(
      snapshot.activeKernelPath,
      snapshot.activeKernelPathSource
    );
    return;
  }

  if (typeof runtimeBridge.setActiveKernelPath === 'function') {
    runtimeBridge.setActiveKernelPath(null, 'none');
  }
}

export async function applyRuntimeInputs(request, runtimeBridge, options = {}) {
  resetRuntimeState(runtimeBridge);

  if (request.runtimePreset) {
    await runtimeBridge.applyRuntimePreset(request.runtimePreset, options);
  }

  if (request.runtimeConfigUrl) {
    await runtimeBridge.applyRuntimeConfigFromUrl(request.runtimeConfigUrl, options);
  }

  mergeRuntimePatch(runtimeBridge, request.runtimeConfig);
  mergeRuntimePatch(runtimeBridge, buildRuntimeContractPatch(request));
}

export async function runWithRuntimeIsolation(runtimeBridge, run) {
  const snapshot = snapshotRuntimeState(runtimeBridge);
  try {
    return await run();
  } finally {
    restoreRuntimeState(runtimeBridge, snapshot);
  }
}

export function buildSuiteOptions(request) {
  return {
    suite: request.suite,
    modelId: request.modelId ?? undefined,
    modelUrl: request.modelUrl ?? undefined,
    cacheMode: request.cacheMode ?? 'warm',
    loadMode: request.loadMode ?? null,
    runtimePreset: request.runtimePreset ?? null,
    captureOutput: request.captureOutput,
    keepPipeline: request.keepPipeline,
    report: request.report || undefined,
    timestamp: request.timestamp ?? undefined,
    searchParams: request.searchParams ?? undefined,
  };
}
