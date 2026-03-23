import {
  applyRuntimeInputs,
  buildSuiteOptions,
  runWithRuntimeIsolation,
} from './command-runner-shared.js';
import { bootstrapNodeWebGPUProvider } from './node-webgpu.js';
import { installNodeFileFetchShim } from './node-file-fetch.js';
import { findFirstDivergence } from '../inference/pipelines/text/operator-events.js';
import { getDriftTolerance } from '../inference/pipelines/text/drift-policy.js';
import { destroyDevice, resetDeviceState } from '../gpu/device.js';
import {
  getActiveKernelPath,
  getActiveKernelPathPolicy,
  getActiveKernelPathSource,
  setActiveKernelPath,
} from '../config/kernel-path-loader.js';

let runtimeModulesPromise = null;

function cloneValue(value) {
  if (value == null) return value;
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

function isPlainObject(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function mergeRecords(base, patch) {
  if (!isPlainObject(base)) {
    return cloneValue(patch);
  }
  if (!isPlainObject(patch)) {
    return cloneValue(base);
  }
  const merged = { ...cloneValue(base) };
  for (const [key, value] of Object.entries(patch)) {
    if (isPlainObject(value) && isPlainObject(merged[key])) {
      merged[key] = mergeRecords(merged[key], value);
      continue;
    }
    merged[key] = cloneValue(value);
  }
  return merged;
}

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

function createRuntimeBridge(modules) {
  return {
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
}

function buildDiagnoseRuntimeConfig(currentConfig) {
  return mergeRecords(currentConfig ?? {}, {
    shared: {
      harness: {
        mode: 'diagnose',
        workload: 'inference',
      },
    },
  });
}

function buildRunSummary(label, provider, response) {
  const metrics = response?.metrics ?? {};
  return {
    label,
    provider,
    modelId: response?.modelId ?? null,
    output: typeof response?.output === 'string' ? response.output : null,
    timing: cloneValue(response?.timing ?? null),
    deviceInfo: cloneValue(response?.deviceInfo ?? null),
    metrics: {
      totalRunMs: metrics.totalRunMs ?? null,
      firstTokenMs: metrics.firstTokenMs ?? null,
      prefillMs: metrics.prefillMs ?? null,
      decodeMs: metrics.decodeMs ?? null,
      decodeTokensPerSec: metrics.decodeTokensPerSec ?? null,
      prefillTokensPerSec: metrics.prefillTokensPerSec ?? null,
      kernelPathId: metrics.kernelPathId ?? null,
      kernelPathSource: metrics.kernelPathSource ?? null,
    },
    operatorDiagnostics: cloneValue(metrics.operatorDiagnostics ?? null),
    reportInfo: cloneValue(response?.reportInfo ?? null),
  };
}

function summarizeDivergence(difference) {
  if (!difference) {
    return {
      found: false,
      type: 'none',
      message: 'No operator divergence detected.',
    };
  }

  if (difference.type === 'within_tolerance') {
    return {
      found: false,
      type: 'within_tolerance',
      opId: difference.opId ?? null,
      opType: difference.opType ?? null,
      tolerance: difference.tolerance ?? null,
      drift: difference.drift ?? null,
      message: 'Operator traces matched within configured drift tolerance.',
    };
  }

  if (difference.type === 'drift_check_needed') {
    return {
      found: false,
      type: 'drift_check_needed',
      opId: difference.opId ?? null,
      opType: difference.opType ?? null,
      tolerance: difference.tolerance ?? null,
      message: 'Operator traces matched structurally, but capture artifacts were insufficient for drift comparison.',
    };
  }

  return {
    found: true,
    ...cloneValue(difference),
  };
}

async function runSingleDiagnostic(modules, request, provider, label) {
  await bootstrapNodeWebGPUProvider(provider, { force: true });
  destroyDevice();
  resetDeviceState();

  const runtimeBridge = createRuntimeBridge(modules);
  try {
    const response = await runWithRuntimeIsolation(runtimeBridge, async () => {
      await applyRuntimeInputs(request, runtimeBridge, {});
      runtimeBridge.setRuntimeConfig(
        buildDiagnoseRuntimeConfig(runtimeBridge.getRuntimeConfig())
      );
      return modules.harness.runBrowserSuite(buildSuiteOptions({
        ...request,
        command: 'diagnose',
        workload: 'inference',
      }, 'node'));
    });

    return buildRunSummary(label, provider, response);
  } finally {
    destroyDevice();
    resetDeviceState();
  }
}

export async function runDiagnoseCommand(request, _options = {}) {
  const modules = await loadRuntimeModules();
  const baselineProvider = request.baselineProvider
    || process.env.DOPPLER_DIAGNOSE_BASELINE_PROVIDER
    || 'webgpu';
  const observedProvider = request.observedProvider
    || process.env.DOPPLER_DIAGNOSE_OBSERVED_PROVIDER
    || '@simulatte/webgpu';

  const baseline = await runSingleDiagnostic(modules, request, baselineProvider, 'baseline');
  const observed = await runSingleDiagnostic(modules, request, observedProvider, 'observed');

  const baselineTimeline = baseline.operatorDiagnostics?.timeline ?? [];
  const observedTimeline = observed.operatorDiagnostics?.timeline ?? [];

  const divergence = baselineTimeline.length > 0 && observedTimeline.length > 0
    ? summarizeDivergence(findFirstDivergence(
      baselineTimeline,
      observedTimeline,
      getDriftTolerance
    ))
    : {
      found: false,
      type: 'missing_timeline',
      message: 'At least one provider run did not emit operator diagnostics.',
      baselineRecords: baseline.operatorDiagnostics?.recordCount ?? 0,
      observedRecords: observed.operatorDiagnostics?.recordCount ?? 0,
    };

  return {
    mode: 'operator_diff',
    baselineProvider,
    observedProvider,
    baseline,
    observed,
    divergence,
  };
}
