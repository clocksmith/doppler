import {
  normalizeToolingCommandRequest,
  ensureCommandSupportedOnSurface,
} from './command-api.js';
import {
  createToolingSuccessEnvelope,
  normalizeToToolingCommandError,
} from './command-envelope.js';
import { assertCommandRequestIsObject, normalizeCommandOptions } from './command-validation.js';
import { convertSafetensorsDirectory } from './node-converter.js';
import { installNodeFileFetchShim } from './node-file-fetch.js';
import { bootstrapNodeWebGPU } from './node-webgpu.js';
import { runDiagnoseCommand } from './diagnose-runner.js';
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

function assertNoUnsupportedRuntimeInputs(request, reason = null) {
  const runtimeFields = [];
  if (Array.isArray(request?.configChain) && request.configChain.length > 0) {
    runtimeFields.push('configChain');
  }
  if (typeof request?.runtimeProfile === 'string' && request.runtimeProfile.trim()) {
    runtimeFields.push('runtimeProfile');
  }
  if (typeof request?.runtimeConfigUrl === 'string' && request.runtimeConfigUrl.trim()) {
    runtimeFields.push('runtimeConfigUrl');
  }
  if (request?.runtimeConfig != null) {
    runtimeFields.push('runtimeConfig');
  }
  if (runtimeFields.length > 0) {
    const reasonSuffix = reason
      ? ` Reason: ${reason}.`
      : ' Put those settings into the workload/config asset instead.';
    throw new Error(
      `${request.command} does not support runtime input fields on the node operator surface: ` +
      `${runtimeFields.join(', ')}.${reasonSuffix}`
    );
  }
}

let runtimeModulesPromise = null;
let trainingOperatorModulesPromise = null;

function loadRuntimeModules() {
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

async function loadTrainingOperatorModules() {
  if (trainingOperatorModulesPromise) {
    return trainingOperatorModulesPromise;
  }
  trainingOperatorModulesPromise = import('../experimental/training/operator-command.js');
  return trainingOperatorModulesPromise;
}

export function hasNodeWebGPUSupport() {
  const hasNavigatorGpu = typeof globalThis.navigator !== 'undefined' && !!globalThis.navigator.gpu;
  const hasGpuEnums = typeof globalThis.GPUBufferUsage !== 'undefined' && typeof globalThis.GPUShaderStage !== 'undefined';
  return hasNavigatorGpu && hasGpuEnums;
}

async function assertNodeWebGPUSupport() {
  const bootstrap = await bootstrapNodeWebGPU();
  const bootstrapProvider = bootstrap.provider ?? null;
  const bootstrapOk = bootstrap.ok === true;
  const bootstrapDetail = typeof bootstrap.detail === 'string' && bootstrap.detail.trim().length > 0
    ? bootstrap.detail.trim()
    : null;

  if (bootstrapOk && hasNodeWebGPUSupport()) return;

  const providerDetail = bootstrapProvider
    ? ` Provider "${bootstrapProvider}" was attempted but did not yield a usable adapter.${bootstrapDetail ? ` Detail: ${bootstrapDetail}.` : ''}`
    : ` No WebGPU provider produced a usable adapter.${bootstrapDetail ? ` Detail: ${bootstrapDetail}.` : ''}`;
  throw new Error(
    'node command: WebGPU runtime is incomplete in Node.' +
    providerDetail +
    ' Run in browser relay, or run under a WebGPU-enabled Node build.'
  );
}

export async function runNodeCommand(commandRequest, options = {}) {
  assertCommandRequestIsObject(commandRequest, 'node');
  const validatedOptions = normalizeCommandOptions(options, 'node');
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
        onProgress: validatedOptions.onProgress,
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
      assertNoUnsupportedRuntimeInputs(request, 'training operator commands resolve runtime from the workload config asset');
      if (!gpuOptionalActions.has(request.action)) {
        await assertNodeWebGPUSupport();
      }
      const training = await loadTrainingOperatorModules();
      const result = await training.runTrainingOperatorCommand(request);
      return createToolingSuccessEnvelope({
        surface: 'node',
        request,
        result,
      });
    }

    if (request.command === 'diagnose') {
      const result = await runDiagnoseCommand(request, validatedOptions);
      return createToolingSuccessEnvelope({
        surface: 'node',
        request,
        result,
      });
    }

    await assertNodeWebGPUSupport();
    const modules = await loadRuntimeModules();
    const runtimeBridge = {
      applyRuntimeProfile: modules.harness.applyRuntimeProfile,
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
      await applyRuntimeInputs(request, runtimeBridge, validatedOptions.runtimeLoadOptions || {});
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
