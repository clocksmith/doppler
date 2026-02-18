import {
  runBrowserSuite,
  applyRuntimePreset,
  applyRuntimeConfigFromUrl,
} from '../inference/browser-harness.js';
import { getRuntimeConfig, setRuntimeConfig } from '../config/runtime.js';
import {
  normalizeToolingCommandRequest,
  buildRuntimeContractPatch,
  ensureCommandSupportedOnSurface,
} from './command-api.js';

function isPlainObject(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function mergeRuntimeValues(base, override) {
  if (override === undefined) return base;
  if (override === null) return null;
  if (!isPlainObject(base) || !isPlainObject(override)) {
    return override;
  }
  const merged = { ...base };
  for (const [key, value] of Object.entries(override)) {
    if (value === undefined) continue;
    merged[key] = mergeRuntimeValues(base[key], value);
  }
  return merged;
}

async function applyRuntimeInputs(request, options = {}) {
  if (request.runtimePreset) {
    await applyRuntimePreset(request.runtimePreset, options);
  }

  if (request.runtimeConfigUrl) {
    await applyRuntimeConfigFromUrl(request.runtimeConfigUrl, options);
  }

  if (request.runtimeConfig) {
    const mergedRuntime = mergeRuntimeValues(getRuntimeConfig(), request.runtimeConfig);
    setRuntimeConfig(mergedRuntime);
  }

  const patch = buildRuntimeContractPatch(request);
  if (!patch) return;

  const mergedRuntime = mergeRuntimeValues(getRuntimeConfig(), patch);
  setRuntimeConfig(mergedRuntime);
}

function buildSuiteOptions(request) {
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

export async function runBrowserCommand(commandRequest, options = {}) {
  const { request } = ensureCommandSupportedOnSurface(commandRequest, 'browser');

  if (request.command === 'convert') {
    if (typeof options.convertHandler !== 'function') {
      throw new Error(
        'browser command convert requires options.convertHandler(request) to be provided.'
      );
    }
    const result = await options.convertHandler(request);
    return {
      ok: true,
      surface: 'browser',
      request,
      result,
    };
  }

  await applyRuntimeInputs(request, options.runtimeLoadOptions || {});
  const result = await runBrowserSuite(buildSuiteOptions(request));

  return {
    ok: true,
    surface: 'browser',
    request,
    result,
  };
}

export function normalizeBrowserCommand(commandRequest) {
  return normalizeToolingCommandRequest(commandRequest);
}
