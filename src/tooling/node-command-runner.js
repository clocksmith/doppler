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
import { runTrainingOperatorCommand } from '../training/operator-command.js';

function asOptionalPlainObject(value, label) {
  if (value == null) return null;
  if (!isPlainObject(value)) {
    throw new Error(`node command: ${label} must be an object when provided.`);
  }
  return value;
}

function assertNoUnsupportedRuntimeInputs(request) {
  const runtimeFields = [];
  if (Array.isArray(request?.configChain) && request.configChain.length > 0) {
    runtimeFields.push('configChain');
  }
  if (typeof request?.runtimePreset === 'string' && request.runtimePreset.trim()) {
    runtimeFields.push('runtimePreset');
  }
  if (typeof request?.runtimeConfigUrl === 'string' && request.runtimeConfigUrl.trim()) {
    runtimeFields.push('runtimeConfigUrl');
  }
  if (request?.runtimeConfig != null) {
    runtimeFields.push('runtimeConfig');
  }
  if (runtimeFields.length > 0) {
    throw new Error(
      `${request.command} does not support runtime input fields on the node operator surface: ` +
      `${runtimeFields.join(', ')}. Put those settings into the workload/config asset instead.`
    );
  }
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
  let bootstrapProvider = null;
  if (!hasNodeWebGPUSupport()) {
    const bootstrap = await bootstrapNodeWebGPU();
    if (bootstrap.ok && bootstrap.provider) {
      bootstrapProvider = bootstrap.provider;
    }
  }

  if (hasNodeWebGPUSupport()) return;
  throw new Error(
    'node command: WebGPU runtime is incomplete in Node.' +
    (bootstrapProvider ? ` Bootstrap attempted provider "${bootstrapProvider}".` : '') +
    ' Run in browser relay, or run under a WebGPU-enabled Node build.'
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

    if (request.command === 'lora' || request.command === 'distill') {
      const gpuOptionalActions = new Set(['compare', 'quality-gate', 'subsets']);
      installNodeFileFetchShim();
      assertNoUnsupportedRuntimeInputs(request);
      if (!gpuOptionalActions.has(request.action)) {
        await assertNodeWebGPUSupport();
      }
      const result = await runTrainingOperatorCommand(request);
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
