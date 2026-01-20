

import { initializeInference, parseRuntimeOverridesFromURL } from './test-harness.js';
import { saveReport } from '../storage/reports.js';
import { getRuntimeConfig, setRuntimeConfig } from '../config/runtime.js';
import { initDevice, getKernelCapabilities, getDevice } from '../gpu/device.js';
import { createPipeline } from './pipeline.js';
import { openModelStore, loadManifestFromStore, loadShard } from '../storage/shard-manager.js';
import { parseManifest } from '../storage/rdrr-format.js';

function resolveRuntime(options) {
  if (options.runtime) return options.runtime;
  if (options.searchParams) return parseRuntimeOverridesFromURL(options.searchParams);
  return parseRuntimeOverridesFromURL();
}

function normalizePresetPath(value) {
  const trimmed = String(value || '').replace(/^[./]+/, '');
  if (!trimmed) return null;
  return trimmed.endsWith('.json') ? trimmed : `${trimmed}.json`;
}

function resolvePresetBaseUrl() {
  try {
    return new URL('../config/presets/runtime/', import.meta.url).toString().replace(/\/$/, '');
  } catch {
    return '/src/config/presets/runtime';
  }
}

function resolveRuntimeFromConfig(config) {
  if (!config || typeof config !== 'object') return null;
  if (config.runtime && typeof config.runtime === 'object') return config.runtime;
  if (config.shared || config.loading || config.inference || config.emulation) return config;
  return null;
}

export async function loadRuntimeConfigFromUrl(url, options = {}) {
  if (!url) {
    throw new Error('runtime config url is required');
  }
  const response = await fetch(url, { signal: options.signal });
  if (!response.ok) {
    throw new Error(`Failed to load runtime config: ${response.status}`);
  }
  const config = await response.json();
  const runtime = resolveRuntimeFromConfig(config);
  if (!runtime) {
    throw new Error('Runtime config is missing runtime fields');
  }
  return { config, runtime };
}

export async function applyRuntimeConfigFromUrl(url, options = {}) {
  const { runtime } = await loadRuntimeConfigFromUrl(url, options);
  setRuntimeConfig(runtime);
  return runtime;
}

export async function loadRuntimePreset(presetId, options = {}) {
  const baseUrl = options.baseUrl || resolvePresetBaseUrl();
  const normalized = normalizePresetPath(presetId);
  if (!normalized) {
    throw new Error('runtime preset id is required');
  }
  const url = `${baseUrl.replace(/\/$/, '')}/${normalized}`;
  return loadRuntimeConfigFromUrl(url, options);
}

export async function applyRuntimePreset(presetId, options = {}) {
  const { runtime } = await loadRuntimePreset(presetId, options);
  setRuntimeConfig(runtime);
  return runtime;
}

export async function initializeBrowserHarness(options = {}) {
  const { modelUrl, onProgress, log } = options;
  if (!modelUrl) {
    throw new Error('modelUrl is required');
  }

  const runtime = resolveRuntime(options);
  const result = await initializeInference(modelUrl, {
    runtime,
    onProgress,
    log,
  });

  return { ...result, runtime };
}

export async function saveBrowserReport(modelId, report, options = {}) {
  return saveReport(modelId, report, options);
}

export async function runBrowserHarness(options = {}) {
  const harness = await initializeBrowserHarness(options);
  const modelId = options.modelId || harness.manifest?.modelId || 'unknown';

  let report = options.report || null;
  if (!report && typeof options.buildReport === 'function') {
    report = await options.buildReport(harness);
  }
  if (!report) {
    report = {
      modelId,
      timestamp: new Date().toISOString(),
    };
  }

  const reportInfo = await saveReport(modelId, report, { timestamp: options.timestamp });
  return { ...harness, report, reportInfo };
}

function normalizeSuite(value) {
  const suite = String(value || '').trim().toLowerCase();
  if (!suite) return 'inference';
  if (suite === 'benchmark') return 'bench';
  return suite;
}

function buildSuiteSummary(suite, results, startTime) {
  let passed = 0;
  let failed = 0;
  let skipped = 0;
  for (const result of results) {
    if (result.skipped) {
      skipped++;
    } else if (result.passed) {
      passed++;
    } else {
      failed++;
    }
  }
  const duration = Math.max(0, performance.now() - startTime);
  return { suite, passed, failed, skipped, duration, results };
}

function buildDeterministicValues(length, scale = 0.01) {
  const data = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    data[i] = Math.sin(i * 0.13) * scale;
  }
  return data;
}

async function initializeInferenceFromStorage(modelId, options = {}) {
  const { onProgress } = options;
  if (!modelId) {
    throw new Error('modelId is required');
  }

  if (options.runtime?.runtimeConfig) {
    setRuntimeConfig(options.runtime.runtimeConfig);
  }

  onProgress?.('storage', 0.05, 'Opening model store...');
  await openModelStore(modelId);

  onProgress?.('manifest', 0.1, 'Loading manifest...');
  const manifestText = await loadManifestFromStore();
  if (!manifestText) {
    throw new Error('Manifest not found in storage');
  }
  const manifest = parseManifest(manifestText);

  onProgress?.('gpu', 0.2, 'Initializing WebGPU...');
  await initDevice();
  const device = getDevice();
  const capabilities = getKernelCapabilities();

  onProgress?.('pipeline', 0.3, 'Creating pipeline...');
  const pipeline = await createPipeline(manifest, {
    gpu: { device },
    storage: { loadShard },
    runtime: options.runtime,
    onProgress,
  });

  return { pipeline, manifest, capabilities };
}

async function initializeSuiteModel(options = {}) {
  const runtime = resolveRuntime(options);
  if (options.modelId && !options.modelUrl) {
    return initializeInferenceFromStorage(options.modelId, { ...options, runtime });
  }
  if (!options.modelUrl) {
    throw new Error('modelUrl is required for this suite');
  }
  return initializeInference(options.modelUrl, {
    runtime,
    onProgress: options.onProgress,
    log: options.log,
  });
}

async function runKernelSuite(options = {}) {
  const startTime = performance.now();
  const results = [];
  const { testHarness, initGPU } = await import('../../tests/kernels/browser/test-page.js');
  await initGPU();

  const gpu = await testHarness.getGPU();
  const checks = [
    {
      name: 'matmul',
      run: async () => {
        if (!testHarness.runMatmul || !testHarness.references?.matmulRef) {
          return { skipped: true };
        }
        const M = 4;
        const N = 4;
        const K = 4;
        const A = buildDeterministicValues(M * K);
        const B = buildDeterministicValues(K * N);
        const expected = testHarness.references.matmulRef(A, B, M, N, K);
        const actual = await testHarness.runMatmul(gpu.device, A, B, M, N, K);
        let maxError = 0;
        for (let i = 0; i < expected.length; i++) {
          maxError = Math.max(maxError, Math.abs(expected[i] - actual[i]));
        }
        if (maxError > 1e-3) {
          throw new Error(`max error ${maxError.toFixed(6)}`);
        }
        return { maxError };
      },
    },
    {
      name: 'rmsnorm',
      run: async () => {
        if (!testHarness.runRMSNorm || !testHarness.references?.rmsNormRef) {
          return { skipped: true };
        }
        const numTokens = 4;
        const hiddenSize = 8;
        const input = buildDeterministicValues(numTokens * hiddenSize);
        const weight = buildDeterministicValues(hiddenSize, 0.02);
        const expected = testHarness.references.rmsNormRef(input, weight, numTokens, hiddenSize, 1e-5);
        const actual = await testHarness.runRMSNorm(gpu.device, input, weight, numTokens, hiddenSize, 1e-5);
        let maxError = 0;
        for (let i = 0; i < expected.length; i++) {
          maxError = Math.max(maxError, Math.abs(expected[i] - actual[i]));
        }
        if (maxError > 1e-3) {
          throw new Error(`max error ${maxError.toFixed(6)}`);
        }
        return { maxError };
      },
    },
  ];

  for (const check of checks) {
    const checkStart = performance.now();
    try {
      const info = await check.run();
      if (info?.skipped) {
        results.push({ name: check.name, passed: false, skipped: true, duration: performance.now() - checkStart });
      } else {
        results.push({ name: check.name, passed: true, duration: performance.now() - checkStart });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      results.push({ name: check.name, passed: false, duration: performance.now() - checkStart, error: message });
    }
  }

  const summary = buildSuiteSummary('kernels', results, startTime);
  return {
    ...summary,
    deviceInfo: getKernelCapabilities(),
  };
}

function resolvePrompt(runtimeConfig, options = {}) {
  if (typeof options.prompt === 'string' && options.prompt.trim()) {
    return options.prompt.trim();
  }
  const runtimePrompt = runtimeConfig?.inference?.prompt;
  if (typeof runtimePrompt === 'string' && runtimePrompt.trim()) {
    return runtimePrompt.trim();
  }
  return 'Hello from Doppler.';
}

function resolveMaxTokens(runtimeConfig, options = {}) {
  if (Number.isFinite(options.maxTokens)) {
    return Math.max(1, Math.floor(options.maxTokens));
  }
  const runtimeMax = runtimeConfig?.inference?.batching?.maxTokens;
  if (Number.isFinite(runtimeMax)) {
    return Math.max(1, Math.floor(runtimeMax));
  }
  return 64;
}

async function runGeneration(pipeline, runtimeConfig, options = {}) {
  const tokens = [];
  const tokenIds = [];
  const prompt = resolvePrompt(runtimeConfig, options);
  const maxTokens = resolveMaxTokens(runtimeConfig, options);
  const sampling = { ...(runtimeConfig.inference?.sampling || {}), ...(options.sampling || {}) };
  const debugProbes = runtimeConfig.shared?.debug?.probes || [];
  const profile = runtimeConfig.shared?.debug?.profiler?.enabled === true;
  const disableCommandBatching = Array.isArray(debugProbes) && debugProbes.length > 0;
  const start = performance.now();

  for await (const tokenText of pipeline.generate(prompt, {
    maxTokens,
    temperature: sampling.temperature,
    topP: sampling.topP,
    topK: sampling.topK,
    repetitionPenalty: sampling.repetitionPenalty,
    greedyThreshold: sampling.greedyThreshold,
    useChatTemplate: false,
    profile,
    disableCommandBatching,
    onToken: (tokenId) => {
      tokenIds.push(tokenId);
    },
  })) {
    if (typeof tokenText === 'string') {
      tokens.push(tokenText);
    }
  }

  const durationMs = Math.max(1, performance.now() - start);
  const tokensPerSec = (tokens.length / durationMs) * 1000;
  return {
    prompt,
    maxTokens,
    tokens,
    tokenIds,
    output: tokens.join(''),
    durationMs,
    tokensPerSec,
  };
}

async function runInferenceSuite(options = {}) {
  const startTime = performance.now();
  const harness = await initializeSuiteModel(options);
  const runtimeConfig = getRuntimeConfig();
  const run = await runGeneration(harness.pipeline, runtimeConfig, options);
  if (typeof harness.pipeline.unload === 'function' && !options.keepPipeline) {
    await harness.pipeline.unload();
  }

  const results = [
    {
      name: 'generation',
      passed: run.tokens.length > 0,
      duration: run.durationMs,
      error: run.tokens.length > 0 ? undefined : 'No tokens generated',
    },
  ];

  const summary = buildSuiteSummary(options.suiteName || 'inference', results, startTime);
  return {
    ...summary,
    modelId: options.modelId || harness.manifest?.modelId || 'unknown',
    output: run.output,
    metrics: {
      prompt: run.prompt,
      maxTokens: run.maxTokens,
      tokensGenerated: run.tokens.length,
      tokensPerSec: Number(run.tokensPerSec.toFixed(2)),
    },
    deviceInfo: getKernelCapabilities(),
  };
}

function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }
  return sorted[mid];
}

async function runBenchSuite(options = {}) {
  const startTime = performance.now();
  const runtimeConfig = getRuntimeConfig();
  const benchConfig = runtimeConfig.shared?.benchmark?.run || {};
  const warmupRuns = Math.max(0, Math.floor(benchConfig.warmupRuns ?? 0));
  const timedRuns = Math.max(1, Math.floor(benchConfig.timedRuns ?? 1));
  const maxTokens = Number.isFinite(benchConfig.maxNewTokens) ? benchConfig.maxNewTokens : undefined;

  const harness = await initializeSuiteModel(options);
  const tokensPerSec = [];
  const durations = [];
  const tokensGenerated = [];

  for (let i = 0; i < warmupRuns + timedRuns; i++) {
    harness.pipeline.reset?.();
    const run = await runGeneration(harness.pipeline, runtimeConfig, {
      ...options,
      maxTokens: maxTokens ?? options.maxTokens,
      sampling: benchConfig.sampling,
    });
    if (i >= warmupRuns) {
      tokensPerSec.push(run.tokensPerSec);
      durations.push(run.durationMs);
      tokensGenerated.push(run.tokens.length);
    }
  }

  if (typeof harness.pipeline.unload === 'function' && !options.keepPipeline) {
    await harness.pipeline.unload();
  }

  const results = [
    {
      name: 'benchmark',
      passed: tokensPerSec.length > 0,
      duration: durations.reduce((sum, value) => sum + value, 0),
      error: tokensPerSec.length > 0 ? undefined : 'No benchmark runs completed',
    },
  ];

  const summary = buildSuiteSummary('bench', results, startTime);
  return {
    ...summary,
    modelId: options.modelId || harness.manifest?.modelId || 'unknown',
    metrics: {
      warmupRuns,
      timedRuns,
      maxTokens: maxTokens ?? resolveMaxTokens(runtimeConfig, options),
      medianTokensPerSec: Number(median(tokensPerSec).toFixed(2)),
      avgTokensPerSec: Number((tokensPerSec.reduce((a, b) => a + b, 0) / (tokensPerSec.length || 1)).toFixed(2)),
      avgTokensGenerated: Math.round(tokensGenerated.reduce((a, b) => a + b, 0) / (tokensGenerated.length || 1)),
    },
    deviceInfo: getKernelCapabilities(),
  };
}

export async function runBrowserSuite(options = {}) {
  const suite = normalizeSuite(options.suite);
  let suiteResult;
  if (suite === 'kernels') {
    suiteResult = await runKernelSuite(options);
  } else if (suite === 'bench') {
    suiteResult = await runBenchSuite(options);
  } else if (suite === 'debug') {
    suiteResult = await runInferenceSuite({ ...options, suiteName: 'debug' });
  } else {
    suiteResult = await runInferenceSuite({ ...options, suiteName: 'inference' });
  }

  const modelId = suiteResult.modelId || options.modelId || options.modelUrl || suite;
  const report = {
    suite,
    modelId,
    runtimePreset: options.runtimePreset ?? null,
    deviceInfo: suiteResult.deviceInfo ?? null,
    results: suiteResult.results,
    durationMs: suiteResult.duration,
    timestamp: new Date().toISOString(),
    metrics: suiteResult.metrics ?? null,
    output: suiteResult.output ?? null,
    ...options.report,
  };
  const reportInfo = await saveReport(modelId, report, { timestamp: options.timestamp });
  return { ...suiteResult, report, reportInfo };
}
