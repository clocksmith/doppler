import {
  normalizeToolingCommandRequest,
  ensureCommandSupportedOnSurface,
} from './command-api.js';
import {
  createToolingSuccessEnvelope,
  normalizeToToolingCommandError,
} from './command-envelope.js';
import { convertSafetensorsDirectory } from './node-converter.js';
import { installNodeFileFetchShim } from './node-file-fetch.js';
import { bootstrapNodeWebGPU } from './node-webgpu.js';
import { applyRuntimeInputs, buildSuiteOptions } from './command-runner-shared.js';
import { runWithRuntimeIsolation } from './command-runner-shared.js';
import { isPlainObject } from '../utils/plain-object.js';
import {
  getActiveKernelPath,
  getActiveKernelPathPolicy,
  getActiveKernelPathSource,
  setActiveKernelPath,
} from '../config/kernel-path-loader.js';

function asOptionalPlainObject(value, label) {
  if (value == null) return null;
  if (!isPlainObject(value)) {
    throw new Error(`node command: ${label} must be an object when provided.`);
  }
  return value;
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
    const bootstrap = await bootstrapNodeWebGPU();
    if (bootstrap.ok && bootstrap.provider) {
      console.error(`[surface] node WebGPU provider: ${bootstrap.provider}`);
    }
  }

  if (hasNodeWebGPUSupport()) return;
  throw new Error(
    'node command: WebGPU runtime is incomplete in Node. Run in browser relay, or run under a WebGPU-enabled Node build.'
  );
}

export async function runNodeCommand(commandRequest, options = {}) {
  let request = null;
  try {
    ({ request } = ensureCommandSupportedOnSurface(commandRequest, 'node'));

    if (request.command === 'convert') {
      const convertPayload = asOptionalPlainObject(request.convertPayload, 'convertPayload');
      const converterConfig = convertPayload
        ? asOptionalPlainObject(convertPayload.converterConfig, 'convertPayload.converterConfig')
        : null;
      const execution = convertPayload
        ? asOptionalPlainObject(convertPayload.execution, 'convertPayload.execution')
        : null;
      const result = await convertSafetensorsDirectory({
        inputDir: request.inputDir,
        outputDir: request.outputDir,
        converterConfig,
        execution,
        onProgress: options.onProgress,
      });
      return createToolingSuccessEnvelope({
        surface: 'node',
        request,
        result,
      });
    }

    await assertNodeWebGPUSupport();
    const modules = await loadRuntimeModules();
    const runtimeBridge = {
      applyRuntimePreset: modules.harness.applyRuntimePreset,
      applyRuntimeConfigFromUrl: modules.harness.applyRuntimeConfigFromUrl,
      getRuntimeConfig: modules.runtime.getRuntimeConfig,
      setRuntimeConfig: modules.runtime.setRuntimeConfig,
      resetRuntimeConfig: modules.runtime.resetRuntimeConfig,
      getActiveKernelPath,
      getActiveKernelPathPolicy,
      getActiveKernelPathSource,
      setActiveKernelPath,
    };

    return runWithRuntimeIsolation(runtimeBridge, async () => {
      await applyRuntimeInputs(request, runtimeBridge, options.runtimeLoadOptions || {});
      const result = await modules.harness.runBrowserSuite(buildSuiteOptions(request, 'node'));

      return createToolingSuccessEnvelope({
        surface: 'node',
        request,
        result,
      });
    });
  } catch (error) {
    throw normalizeToToolingCommandError(error, {
      surface: 'node',
      request,
    });
  }
}

export function normalizeNodeCommand(commandRequest) {
  return normalizeToolingCommandRequest(commandRequest);
}
