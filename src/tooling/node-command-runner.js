import {
  normalizeToolingCommandRequest,
  buildRuntimeContractPatch,
  ensureCommandSupportedOnSurface,
} from './command-api.js';
import { convertSafetensorsDirectory } from './node-convert.js';
import { installNodeFileFetchShim } from './node-file-fetch.js';
import { bootstrapNodeWebGPU } from './node-webgpu.js';

function isPlainObject(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function asOptionalPlainObject(value, label) {
  if (value == null) return null;
  if (!isPlainObject(value)) {
    throw new Error(`node command: ${label} must be an object when provided.`);
  }
  return value;
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

let runtimeModulesPromise = null;

async function loadRuntimeModules() {
  if (runtimeModulesPromise) {
    return runtimeModulesPromise;
  }

  installNodeFileFetchShim();
  runtimeModulesPromise = Promise.all([
    import('../inference/browser-harness.js'),
    import('../config/runtime.js'),
  ]).then(([harness, runtime]) => ({ harness, runtime }));

  return runtimeModulesPromise;
}

export function hasNodeWebGPUSupport() {
  const hasNavigatorGpu = typeof globalThis.navigator !== 'undefined' && !!globalThis.navigator.gpu;
  const hasGpuEnums = typeof globalThis.GPUBufferUsage !== 'undefined' && typeof globalThis.GPUShaderStage !== 'undefined';
  return hasNavigatorGpu && hasGpuEnums;
}

async function assertNodeWebGPUSupport() {
  if (!hasNodeWebGPUSupport()) {
    await bootstrapNodeWebGPU();
  }

  if (hasNodeWebGPUSupport()) return;
  throw new Error(
    'node command: WebGPU runtime is incomplete in Node. Run in browser relay, or run under a WebGPU-enabled Node build.'
  );
}


async function applyRuntimeInputs(request, modules, options = {}) {
  const { harness, runtime } = modules;

  if (request.runtimePreset) {
    await harness.applyRuntimePreset(request.runtimePreset, options);
  }

  if (request.runtimeConfigUrl) {
    await harness.applyRuntimeConfigFromUrl(request.runtimeConfigUrl, options);
  }

  if (request.runtimeConfig) {
    runtime.setRuntimeConfig(request.runtimeConfig);
  }

  const patch = buildRuntimeContractPatch(request);
  if (!patch) return;

  const mergedRuntime = mergeRuntimeValues(runtime.getRuntimeConfig(), patch);
  runtime.setRuntimeConfig(mergedRuntime);
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

export async function runNodeCommand(commandRequest, options = {}) {
  const { request } = ensureCommandSupportedOnSurface(commandRequest, 'node');

  if (request.command === 'convert') {
    const convertPayload = asOptionalPlainObject(request.convertPayload, 'convertPayload');
    const converterConfig = convertPayload
      ? asOptionalPlainObject(convertPayload.converterConfig, 'convertPayload.converterConfig')
      : null;
    const result = await convertSafetensorsDirectory({
      inputDir: request.inputDir,
      outputDir: request.outputDir,
      modelId: request.modelId,
      converterConfig,
      onProgress: options.onProgress,
    });
    return {
      ok: true,
      surface: 'node',
      request,
      result,
    };
  }

  await assertNodeWebGPUSupport();
  const modules = await loadRuntimeModules();
  await applyRuntimeInputs(request, modules, options.runtimeLoadOptions || {});
  const result = await modules.harness.runBrowserSuite(buildSuiteOptions(request));

  return {
    ok: true,
    surface: 'node',
    request,
    result,
  };
}

export function normalizeNodeCommand(commandRequest) {
  return normalizeToolingCommandRequest(commandRequest);
}
