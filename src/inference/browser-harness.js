

import { initializeInference, parseRuntimeOverridesFromURL } from './test-harness.js';
import { saveReport } from '../storage/reports.js';
import { getRuntimeConfig, setRuntimeConfig } from '../config/runtime.js';
import { initDevice, getKernelCapabilities, getDevice } from '../gpu/device.js';
import { createPipeline } from './pipeline.js';
import { parseModelConfigFromManifest } from './pipeline/config.js';
import { openModelStore, loadManifestFromStore, loadShard } from '../storage/shard-manager.js';
import { parseManifest } from '../storage/rdrr-format.js';
import { computeSampleStats } from '../debug/stats.js';
import {
  applyKernelOverrides,
  resolveKernelPath,
  setActiveKernelPath,
  getActiveKernelPath,
  getActiveKernelPathSource,
} from '../config/kernel-path-loader.js';

function resolveRuntime(options) {
  if (options.runtime) return options.runtime;
  if (options.searchParams) return parseRuntimeOverridesFromURL(options.searchParams);
  if (typeof window === 'undefined') return parseRuntimeOverridesFromURL(new URLSearchParams());
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
    if (typeof window !== 'undefined' && window.location?.href) {
      return new URL('/src/config/presets/runtime/', window.location.href).toString().replace(/\/$/, '');
    }
    return '/src/config/presets/runtime';
  }
}

function resolveRuntimeFromConfig(config) {
  if (!config || typeof config !== 'object') return null;
  if (config.runtime && typeof config.runtime === 'object') return config.runtime;
  if (config.shared || config.loading || config.inference || config.emulation) return config;
  return null;
}

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

function normalizeExtends(value) {
  if (Array.isArray(value)) {
    return value.map((entry) => String(entry || '').trim()).filter(Boolean);
  }
  if (typeof value === 'string') {
    const trimmed = value.trim();
    return trimmed ? [trimmed] : [];
  }
  return [];
}

function normalizeExtendsPath(value) {
  const trimmed = String(value || '').trim();
  if (!trimmed) return null;
  return trimmed.endsWith('.json') ? trimmed : `${trimmed}.json`;
}

function resolveAbsoluteUrl(target, base) {
  try {
    if (base) {
      return new URL(target, base).toString();
    }
    if (typeof window !== 'undefined' && window.location?.href) {
      return new URL(target, window.location.href).toString();
    }
    return new URL(target, import.meta.url).toString();
  } catch {
    return target;
  }
}

function isAbsoluteUrl(value) {
  return /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(value);
}

function joinUrl(base, path) {
  if (!base) return path;
  if (isAbsoluteUrl(base)) {
    return new URL(path, base.endsWith('/') ? base : `${base}/`).toString();
  }
  const normalizedBase = base.replace(/\/$/, '');
  const normalizedPath = path.replace(/^\//, '');
  return `${normalizedBase}/${normalizedPath}`;
}

function resolveExtendCandidates(ref, context) {
  const normalized = normalizeExtendsPath(ref);
  if (!normalized) return [];
  if (isAbsoluteUrl(normalized) || normalized.startsWith('/')) {
    return [normalized];
  }
  if (normalized.startsWith('./') || normalized.startsWith('../')) {
    return [resolveAbsoluteUrl(normalized, context.sourceUrl)];
  }
  if (normalized.includes('/')) {
    return [joinUrl(context.presetBaseUrl, normalized)];
  }
  const candidates = [];
  if (context.presetBaseUrl) {
    candidates.push(joinUrl(context.presetBaseUrl, normalized));
    candidates.push(joinUrl(context.presetBaseUrl, `modes/${normalized}`));
  }
  if (context.sourceUrl) {
    const sourceDir = resolveAbsoluteUrl('./', context.sourceUrl);
    candidates.push(resolveAbsoluteUrl(normalized, sourceDir));
  }
  return [...new Set(candidates)];
}

async function fetchRuntimeConfig(url, options = {}) {
  const response = await fetch(url, { signal: options.signal });
  if (!response.ok) {
    const error = new Error(`Failed to load runtime config: ${response.status}`);
    error.code = response.status === 404 ? 'runtime_config_not_found' : 'runtime_config_fetch_failed';
    throw error;
  }
  return response.json();
}

async function resolveRuntimeConfigExtends(config, context) {
  const runtime = resolveRuntimeFromConfig(config);
  if (!runtime) {
    throw new Error('Runtime config is missing runtime fields');
  }

  const extendsRefs = normalizeExtends(config.extends);
  let mergedRuntime = null;
  let mergedConfig = null;

  for (const ref of extendsRefs) {
    const base = await loadRuntimeConfigFromRef(ref, context);
    mergedRuntime = mergedRuntime ? mergeRuntimeValues(mergedRuntime, base.runtime) : base.runtime;
    mergedConfig = mergedConfig ? mergeRuntimeValues(mergedConfig, base.config) : base.config;
  }

  const combinedRuntime = mergedRuntime ? mergeRuntimeValues(mergedRuntime, runtime) : runtime;
  const combinedConfig = mergedConfig ? mergeRuntimeValues(mergedConfig, config) : { ...config };
  const resolved = { ...combinedConfig, runtime: combinedRuntime };
  if (resolved.extends !== undefined) {
    delete resolved.extends;
  }
  return { config: resolved, runtime: combinedRuntime };
}

async function loadRuntimeConfigChain(url, options = {}, stack = []) {
  const presetBaseUrl = options.presetBaseUrl || options.baseUrl || resolvePresetBaseUrl();
  const resolvedUrl = resolveAbsoluteUrl(url);
  if (stack.includes(resolvedUrl)) {
    throw new Error(`Runtime config extends cycle: ${[...stack, resolvedUrl].join(' -> ')}`);
  }
  const config = await fetchRuntimeConfig(resolvedUrl, options);
  return resolveRuntimeConfigExtends(config, {
    ...options,
    sourceUrl: resolvedUrl,
    presetBaseUrl,
    stack: [...stack, resolvedUrl],
  });
}

async function loadRuntimeConfigFromRef(ref, context) {
  const candidates = resolveExtendCandidates(ref, context);
  if (!candidates.length) {
    throw new Error(`Runtime config extends is invalid: ${ref}`);
  }
  let lastError = null;
  for (const candidate of candidates) {
    try {
      return await loadRuntimeConfigChain(candidate, context, context.stack ?? []);
    } catch (error) {
      if (error?.code === 'runtime_config_not_found') {
        lastError = error;
        continue;
      }
      throw error;
    }
  }
  if (lastError) {
    throw lastError;
  }
  throw new Error(`Runtime config extends not found: ${ref}`);
}

export async function loadRuntimeConfigFromUrl(url, options = {}) {
  if (!url) {
    throw new Error('runtime config url is required');
  }
  return loadRuntimeConfigChain(url, options);
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
  return loadRuntimeConfigFromUrl(url, { ...options, presetBaseUrl: baseUrl });
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

async function resolveKernelPathForModel(options = {}) {
  const runtimeConfig = options.runtime?.runtimeConfig ?? getRuntimeConfig();
  const runtimeKernelPath = options.runtime?.kernelPath ?? null;
  let manifest = null;
  let manifestModelId = options.modelId || null;

  if (options.modelId) {
    await openModelStore(options.modelId);
    const manifestText = await loadManifestFromStore();
    if (manifestText) {
      manifest = parseManifest(manifestText);
      manifestModelId = manifest.modelId ?? options.modelId;
    }
  }

  if (!manifest) return null;

  const modelConfig = parseModelConfigFromManifest(manifest, runtimeConfig);
  const kernelPathRef = runtimeKernelPath
    ?? runtimeConfig?.inference?.kernelPath
    ?? modelConfig?.kernelPath
    ?? manifest.optimizations?.kernelPath;

  if (!kernelPathRef) {
    setActiveKernelPath(null, 'none');
    return { modelId: manifestModelId, kernelPath: null, source: 'none' };
  }

  let resolved = resolveKernelPath(kernelPathRef);
  if (runtimeConfig?.inference?.kernelOverrides) {
    resolved = applyKernelOverrides(resolved, runtimeConfig.inference.kernelOverrides);
  }
  const source = runtimeKernelPath
    ? 'runtime'
    : runtimeConfig?.inference?.kernelPath
      ? 'config'
      : modelConfig?.kernelPath
        ? 'model'
        : 'manifest';
  setActiveKernelPath(resolved, source);
  return { modelId: manifestModelId, kernelPath: resolved, source };
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
  const { testHarness, initGPU } = await import('../../tests/kernels/browser/test-page.js');
  const { runKernelSuite: runAllKernelTests } = await import('../../tests/kernels/browser/kernel-suite.js');
  await initGPU();

  const previousKernelPath = getActiveKernelPath();
  const previousKernelSource = getActiveKernelPathSource();
  if (options.modelId) {
    await resolveKernelPathForModel(options);
  }
  let results = [];
  try {
    results = await runAllKernelTests(testHarness);
  } finally {
    setActiveKernelPath(previousKernelPath, previousKernelSource);
  }

  const summary = buildSuiteSummary('kernels', results, startTime);
  return {
    ...summary,
    deviceInfo: getKernelCapabilities(),
  };
}

function resolvePrompt(runtimeConfig) {
  const runtimePrompt = runtimeConfig?.inference?.prompt;
  if (typeof runtimePrompt === 'string' && runtimePrompt.trim()) {
    return runtimePrompt.trim();
  }
  throw new Error('runtime.inference.prompt must be set for harness runs.');
}

function resolveMaxTokens(runtimeConfig) {
  const runtimeMax = runtimeConfig?.inference?.batching?.maxTokens;
  if (Number.isFinite(runtimeMax)) {
    return Math.max(1, Math.floor(runtimeMax));
  }
  throw new Error('runtime.inference.batching.maxTokens must be set for harness runs.');
}

async function runGeneration(pipeline, runtimeConfig) {
  const tokens = [];
  const tokenIds = [];
  const prompt = resolvePrompt(runtimeConfig);
  const maxTokens = resolveMaxTokens(runtimeConfig);
  const sampling = runtimeConfig.inference?.sampling || {};
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
  const run = await runGeneration(harness.pipeline, runtimeConfig);
  const memoryStats = typeof harness.pipeline?.getMemoryStats === 'function'
    ? harness.pipeline.getMemoryStats()
    : null;
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
    memoryStats,
    deviceInfo: getKernelCapabilities(),
    pipeline: options.keepPipeline ? harness.pipeline : null,
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
  const benchSampling = isPlainObject(benchConfig.sampling) ? benchConfig.sampling : null;
  const benchOverrides = {};
  if (Number.isFinite(maxTokens)) {
    benchOverrides.inference = { batching: { maxTokens } };
  }
  if (benchSampling) {
    benchOverrides.inference = {
      ...(benchOverrides.inference || {}),
      sampling: benchSampling,
    };
  }
  const benchRuntime = Object.keys(benchOverrides).length > 0
    ? mergeRuntimeValues(runtimeConfig, benchOverrides)
    : runtimeConfig;

  const harness = await initializeSuiteModel(options);
  const tokensPerSec = [];
  const durations = [];
  const tokensGenerated = [];

  for (let i = 0; i < warmupRuns + timedRuns; i++) {
    harness.pipeline.reset?.();
    const run = await runGeneration(harness.pipeline, benchRuntime);
    if (i >= warmupRuns) {
      tokensPerSec.push(run.tokensPerSec);
      durations.push(run.durationMs);
      tokensGenerated.push(run.tokens.length);
    }
  }

  const memoryStats = typeof harness.pipeline?.getMemoryStats === 'function'
    ? harness.pipeline.getMemoryStats()
    : null;

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
      maxTokens: resolveMaxTokens(benchRuntime),
      medianTokensPerSec: Number(median(tokensPerSec).toFixed(2)),
      avgTokensPerSec: Number((tokensPerSec.reduce((a, b) => a + b, 0) / (tokensPerSec.length || 1)).toFixed(2)),
      avgTokensGenerated: Math.round(tokensGenerated.reduce((a, b) => a + b, 0) / (tokensGenerated.length || 1)),
    },
    memoryStats,
    deviceInfo: getKernelCapabilities(),
    pipeline: options.keepPipeline ? harness.pipeline : null,
  };
}

async function runDiffusionSuite(options = {}) {
  const startTime = performance.now();
  const runtimeConfig = getRuntimeConfig();
  const captureOutput = options.captureOutput === true;
  const benchConfig = runtimeConfig.shared?.benchmark?.run || {};
  const warmupRuns = Math.max(0, Math.floor(benchConfig.warmupRuns ?? 0));
  const timedRuns = Math.max(1, Math.floor(benchConfig.timedRuns ?? 1));

  const diffusionConfig = runtimeConfig.inference?.diffusion;
  if (!diffusionConfig) {
    throw new Error('runtime.inference.diffusion must be set for diffusion harness runs.');
  }
  const scheduler = diffusionConfig.scheduler;
  const latent = diffusionConfig.latent;
  const prompt = resolvePrompt(runtimeConfig);
  const negativePrompt = diffusionConfig.negativePrompt ?? '';

  const width = Math.floor(latent?.width);
  const height = Math.floor(latent?.height);
  const steps = Math.floor(scheduler?.numSteps);
  const guidanceScale = scheduler?.guidanceScale;

  if (!Number.isFinite(width) || width <= 0) {
    throw new Error('runtime.inference.diffusion.latent.width must be set for diffusion harness runs.');
  }
  if (!Number.isFinite(height) || height <= 0) {
    throw new Error('runtime.inference.diffusion.latent.height must be set for diffusion harness runs.');
  }
  if (!Number.isFinite(steps) || steps <= 0) {
    throw new Error('runtime.inference.diffusion.scheduler.numSteps must be set for diffusion harness runs.');
  }
  if (!Number.isFinite(guidanceScale) || guidanceScale <= 0) {
    throw new Error('runtime.inference.diffusion.scheduler.guidanceScale must be set for diffusion harness runs.');
  }

  const harness = await initializeSuiteModel(options);
  const totalMs = [];
  const prefillMs = [];
  const denoiseMs = [];
  const vaeMs = [];
  const prefillTokens = [];
  const decodeTokens = [];
  const gpuTotalMs = [];
  const gpuPrefillMs = [];
  const gpuDenoiseMs = [];
  const gpuVaeMs = [];
  let output = null;

  for (let i = 0; i < warmupRuns + timedRuns; i++) {
    harness.pipeline.reset?.();
    const result = await harness.pipeline.generate({
      prompt,
      negativePrompt,
      steps,
      guidanceScale,
      width,
      height,
    });
    if (captureOutput && i === warmupRuns + timedRuns - 1) {
      output = result;
    }

    if (i < warmupRuns) continue;

    const stats = harness.pipeline.getStats?.() ?? {};
    if (Number.isFinite(stats.totalTimeMs)) totalMs.push(stats.totalTimeMs);
    if (Number.isFinite(stats.prefillTimeMs)) prefillMs.push(stats.prefillTimeMs);
    if (Number.isFinite(stats.decodeTimeMs)) denoiseMs.push(stats.decodeTimeMs);
    if (Number.isFinite(stats.vaeTimeMs)) vaeMs.push(stats.vaeTimeMs);
    if (Number.isFinite(stats.prefillTokens)) prefillTokens.push(stats.prefillTokens);
    if (Number.isFinite(stats.decodeTokens)) decodeTokens.push(stats.decodeTokens);

    const gpu = stats.gpu ?? null;
    if (gpu?.available) {
      if (Number.isFinite(gpu.totalMs)) gpuTotalMs.push(gpu.totalMs);
      if (Number.isFinite(gpu.prefillMs)) gpuPrefillMs.push(gpu.prefillMs);
      if (Number.isFinite(gpu.denoiseMs)) gpuDenoiseMs.push(gpu.denoiseMs);
      if (Number.isFinite(gpu.vaeMs)) gpuVaeMs.push(gpu.vaeMs);
    }
  }

  const memoryStats = typeof harness.pipeline?.getMemoryStats === 'function'
    ? harness.pipeline.getMemoryStats()
    : null;

  if (typeof harness.pipeline.unload === 'function' && !options.keepPipeline) {
    await harness.pipeline.unload();
  }

  const results = [
    {
      name: 'diffusion',
      passed: totalMs.length > 0,
      duration: totalMs.reduce((sum, value) => sum + value, 0),
      error: totalMs.length > 0 ? undefined : 'No diffusion runs completed',
    },
  ];

  const summary = buildSuiteSummary('diffusion', results, startTime);
  const cpuStats = {
    totalMs: computeSampleStats(totalMs),
    prefillMs: computeSampleStats(prefillMs),
    denoiseMs: computeSampleStats(denoiseMs),
    vaeMs: computeSampleStats(vaeMs),
  };
  const gpuStats = gpuTotalMs.length > 0
    ? {
        available: true,
        totalMs: computeSampleStats(gpuTotalMs),
        prefillMs: computeSampleStats(gpuPrefillMs),
        denoiseMs: computeSampleStats(gpuDenoiseMs),
        vaeMs: computeSampleStats(gpuVaeMs),
      }
    : { available: false };

  const avgPrefillTokens = prefillTokens.length
    ? Math.round(prefillTokens.reduce((a, b) => a + b, 0) / prefillTokens.length)
    : 0;
  const avgDecodeTokens = decodeTokens.length
    ? Math.round(decodeTokens.reduce((a, b) => a + b, 0) / decodeTokens.length)
    : 0;

  return {
    ...summary,
    modelId: options.modelId || harness.manifest?.modelId || 'unknown',
    output,
    metrics: {
      warmupRuns,
      timedRuns,
      width,
      height,
      steps,
      guidanceScale,
      prompt,
      avgPrefillTokens,
      avgDecodeTokens,
      cpu: cpuStats,
      gpu: gpuStats,
    },
    memoryStats,
    deviceInfo: getKernelCapabilities(),
    pipeline: options.keepPipeline ? harness.pipeline : null,
  };
}

async function runEnergySuite(options = {}) {
  const startTime = performance.now();
  const harness = await initializeSuiteModel(options);
  if (harness.manifest?.modelType !== 'energy') {
    throw new Error('Energy suite requires an energy model manifest.');
  }

  const result = await harness.pipeline.generate();
  const stats = harness.pipeline.getStats?.() ?? {};

  const memoryStats = typeof harness.pipeline?.getMemoryStats === 'function'
    ? harness.pipeline.getMemoryStats()
    : null;

  if (typeof harness.pipeline.unload === 'function' && !options.keepPipeline) {
    await harness.pipeline.unload();
  }

  const results = [
    {
      name: 'energy',
      passed: Number.isFinite(result.energy ?? NaN),
      duration: result.totalTimeMs ?? Math.max(0, performance.now() - startTime),
      error: Number.isFinite(result.energy ?? NaN) ? undefined : 'Energy did not converge',
    },
  ];

  const summary = buildSuiteSummary('energy', results, startTime);
  return {
    ...summary,
    modelId: options.modelId || harness.manifest?.modelId || 'unknown',
    metrics: {
      steps: result.steps,
      energy: result.energy ?? null,
      dtype: result.dtype,
      shape: result.shape,
      totalTimeMs: result.totalTimeMs ?? null,
      energyHistory: result.energyHistory ?? [],
      stateStats: result.stateStats ?? null,
      readbackCount: stats.readbackCount ?? null,
    },
    memoryStats,
    deviceInfo: getKernelCapabilities(),
    pipeline: options.keepPipeline ? harness.pipeline : null,
  };
}

export async function runBrowserSuite(options = {}) {
  const suite = normalizeSuite(options.suite);
  let suiteResult;
  if (suite === 'kernels') {
    suiteResult = await runKernelSuite(options);
  } else if (suite === 'bench') {
    suiteResult = await runBenchSuite(options);
  } else if (suite === 'diffusion') {
    suiteResult = await runDiffusionSuite(options);
  } else if (suite === 'energy') {
    suiteResult = await runEnergySuite(options);
  } else if (suite === 'debug') {
    suiteResult = await runInferenceSuite({ ...options, suiteName: 'debug' });
  } else {
    suiteResult = await runInferenceSuite({ ...options, suiteName: 'inference' });
  }

  const modelId = suiteResult.modelId || options.modelId || options.modelUrl || suite;
  const reportOutput = typeof suiteResult.output === 'string' ? suiteResult.output : null;
  const report = {
    suite,
    modelId,
    runtimePreset: options.runtimePreset ?? null,
    deviceInfo: suiteResult.deviceInfo ?? null,
    results: suiteResult.results,
    durationMs: suiteResult.duration,
    timestamp: new Date().toISOString(),
    metrics: suiteResult.metrics ?? null,
    output: reportOutput,
    memory: suiteResult.memoryStats ?? null,
    ...options.report,
  };
  const reportInfo = await saveReport(modelId, report, { timestamp: options.timestamp });
  return { ...suiteResult, report, reportInfo };
}

function normalizeManifest(manifest) {
  if (!manifest || typeof manifest !== 'object') {
    throw new Error('Harness manifest must be an object.');
  }
  const runs = Array.isArray(manifest.runs) ? manifest.runs : [];
  if (!runs.length) {
    throw new Error('Harness manifest must include at least one run.');
  }
  return {
    defaults: manifest.defaults ?? {},
    runs,
    reportModelId: manifest.reportModelId ?? manifest.id ?? 'manifest',
    report: manifest.report ?? null,
  };
}

function mergeRunDefaults(defaults, run) {
  return {
    ...defaults,
    ...run,
    runtimePreset: run.runtimePreset ?? defaults.runtimePreset ?? null,
    runtimeConfigUrl: run.runtimeConfigUrl ?? defaults.runtimeConfigUrl ?? null,
    runtimeConfig: run.runtimeConfig ?? defaults.runtimeConfig ?? null,
    suite: run.suite ?? defaults.suite ?? 'inference',
  };
}

async function applyRuntimeForRun(run, options) {
  if (run.runtimeConfig) {
    const runtime = resolveRuntimeFromConfig(run.runtimeConfig);
    if (!runtime) {
      throw new Error('runtimeConfig is missing runtime fields');
    }
    setRuntimeConfig(runtime);
    return;
  }
  if (run.runtimeConfigUrl) {
    await applyRuntimeConfigFromUrl(run.runtimeConfigUrl, options);
    return;
  }
  if (run.runtimePreset) {
    await applyRuntimePreset(run.runtimePreset, options);
  }
}

function summarizeManifestRuns(results) {
  let passedRuns = 0;
  let failedRuns = 0;
  let durationMs = 0;
  for (const result of results) {
    const failures = (result.results || []).filter((entry) => !entry.passed && !entry.skipped);
    if (failures.length > 0) {
      failedRuns += 1;
    } else {
      passedRuns += 1;
    }
    durationMs += result.duration || 0;
  }
  return {
    totalRuns: results.length,
    passedRuns,
    failedRuns,
    durationMs,
  };
}

export async function runBrowserManifest(manifest, options = {}) {
  const normalized = normalizeManifest(manifest);
  const results = [];

  for (let i = 0; i < normalized.runs.length; i++) {
    const run = mergeRunDefaults(normalized.defaults, normalized.runs[i] || {});
    await applyRuntimeForRun(run, options);
    const result = await runBrowserSuite(run);
    results.push({
      ...result,
      label: run.label ?? `${run.suite || 'inference'}:${result.modelId || 'unknown'}`,
    });
    options.onProgress?.({
      index: i + 1,
      total: normalized.runs.length,
      label: run.label ?? result.modelId ?? run.suite ?? 'run',
    });
  }

  const summary = summarizeManifestRuns(results);
  const report = {
    timestamp: new Date().toISOString(),
    summary,
    runs: results.map((result) => ({
      label: result.label,
      suite: result.suite,
      modelId: result.modelId,
      results: result.results,
      metrics: result.metrics ?? null,
      output: typeof result.output === 'string' ? result.output : null,
      reportInfo: result.reportInfo ?? null,
    })),
    manifest: normalized.report ?? null,
  };

  const reportInfo = options.saveReport === false
    ? null
    : await saveReport(normalized.reportModelId, report, { timestamp: options.timestamp });

  return { results, summary, report, reportInfo };
}
