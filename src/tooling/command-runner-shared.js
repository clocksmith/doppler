import { buildRuntimeContractPatch } from './command-api.js';
import { mergeRuntimeValues } from '../config/runtime-merge.js';

function mergeRuntimePatch(runtimeBridge, patch) {
  if (!patch) return;
  const mergedRuntime = mergeRuntimeValues(runtimeBridge.getRuntimeConfig(), patch);
  runtimeBridge.setRuntimeConfig(mergedRuntime);
}

export async function applyRuntimeInputs(request, runtimeBridge, options = {}) {
  if (request.runtimePreset) {
    await runtimeBridge.applyRuntimePreset(request.runtimePreset, options);
  }

  if (request.runtimeConfigUrl) {
    await runtimeBridge.applyRuntimeConfigFromUrl(request.runtimeConfigUrl, options);
  }

  mergeRuntimePatch(runtimeBridge, request.runtimeConfig);
  mergeRuntimePatch(runtimeBridge, buildRuntimeContractPatch(request));
}

export function buildSuiteOptions(request) {
  return {
    suite: request.suite,
    modelId: request.modelId ?? undefined,
    modelUrl: request.modelUrl ?? undefined,
    runtimePreset: request.runtimePreset ?? null,
    captureOutput: request.captureOutput,
    keepPipeline: request.keepPipeline,
    report: request.report || undefined,
    timestamp: request.timestamp ?? undefined,
    searchParams: request.searchParams ?? undefined,
  };
}
