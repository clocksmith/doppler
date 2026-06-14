import { getRuntimeConfig } from '../config/runtime.js';
import { computeSampleStats } from '../debug/stats.js';
import { initializeSuiteModel, resolveDeviceInfo } from './browser-harness-model-helpers.js';
import { buildSuiteContractMetrics } from './browser-harness-contract-helpers.js';
import { resolvePrompt } from './browser-harness-text-helpers.js';
import {
  buildSuiteSummary,
  normalizeCacheMode,
  normalizeLoadMode,
  safeStatsValue,
  buildDiffusionPerformanceArtifact,
  buildCanonicalTiming,
  buildTimingDiagnostics,
} from './browser-harness-suite-helpers.js';

function isDiffusionGemmaManifest(manifest) {
  return manifest?.modelType === 'diffusion_gemma';
}

function readOptionalPositiveInteger(source, field, label) {
  const value = source?.[field];
  if (value == null) return null;
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${label}.${field} must be a positive integer when provided.`);
  }
  return value;
}

function readOptionalInteger(source, field, label) {
  const value = source?.[field];
  if (value == null) return null;
  if (!Number.isInteger(value)) {
    throw new Error(`${label}.${field} must be an integer when provided.`);
  }
  return value;
}

function resolveDiffusionGemmaRunConfig(runtimeConfig, manifest) {
  const label = 'runtime.inference.diffusionGemma';
  const source = runtimeConfig?.inference?.diffusionGemma ?? null;
  if (source != null && (typeof source !== 'object' || Array.isArray(source))) {
    throw new Error(`${label} must be an object when provided.`);
  }
  const contract = manifest?.inference?.diffusionGemma ?? null;
  const maxNewTokens = readOptionalPositiveInteger(source, 'maxNewTokens', label)
    ?? contract?.maxNewTokens
    ?? null;
  const canvasLength = readOptionalPositiveInteger(source, 'canvasLength', label)
    ?? contract?.canvasLength
    ?? null;
  const maxDenoisingSteps = readOptionalPositiveInteger(source, 'maxDenoisingSteps', label)
    ?? contract?.maxDenoisingSteps
    ?? null;
  const seed = readOptionalInteger(source, 'seed', label);
  return {
    maxNewTokens,
    canvasLength,
    maxDenoisingSteps,
    seed,
  };
}

function decodeDiffusionGemmaOutput(pipeline, tokenIds) {
  const ids = Array.from(tokenIds);
  if (typeof pipeline?.tokenizer?.decode === 'function') {
    return pipeline.tokenizer.decode(ids);
  }
  return ids.join(' ');
}

function averageRounded(values) {
  if (!values.length) return 0;
  return Math.round(values.reduce((sum, value) => sum + value, 0) / values.length);
}

async function runDiffusionGemmaSuite(options, context) {
  const {
    startTime,
    runtimeConfig,
    captureOutput,
    cacheMode,
    loadMode,
    warmupRuns,
    timedRuns,
    harness,
  } = context;
  const prompt = resolvePrompt(runtimeConfig);
  const runConfig = resolveDiffusionGemmaRunConfig(runtimeConfig, harness.manifest);
  const totalMs = [];
  const prefillMs = [];
  const denoiseMs = [];
  const vaeMs = [];
  const prefillTokens = [];
  const decodeTokens = [];
  const denoiseSteps = [];
  const tokensPerForward = [];
  let output = null;

  for (let i = 0; i < warmupRuns + timedRuns; i += 1) {
    harness.pipeline.reset?.();
    const generateOptions = {};
    if (runConfig.maxNewTokens != null) {
      generateOptions.maxNewTokens = runConfig.maxNewTokens;
    }
    if (runConfig.seed != null) {
      generateOptions.seed = runConfig.seed;
    }
    if (runConfig.canvasLength != null) {
      generateOptions.canvasLength = runConfig.canvasLength;
    }
    if (runConfig.maxDenoisingSteps != null) {
      generateOptions.maxDenoisingSteps = runConfig.maxDenoisingSteps;
    }
    const runStart = performance.now();
    const tokenIds = await harness.pipeline.generateTokenIds(prompt, generateOptions);
    const wallMs = Math.max(0, performance.now() - runStart);
    const stats = harness.pipeline.getStats?.() ?? {};

    if (captureOutput && i === warmupRuns + timedRuns - 1) {
      output = {
        tokenIds: Array.from(tokenIds),
        text: decodeDiffusionGemmaOutput(harness.pipeline, tokenIds),
      };
    }

    if (i < warmupRuns) continue;

    totalMs.push(Number.isFinite(stats.totalTimeMs) ? stats.totalTimeMs : wallMs);
    prefillMs.push(Number.isFinite(stats.prefillTimeMs) ? stats.prefillTimeMs : 0);
    denoiseMs.push(Number.isFinite(stats.decodeTimeMs) ? stats.decodeTimeMs : wallMs);
    vaeMs.push(0);
    prefillTokens.push(Number.isFinite(stats.prefillTokens) ? stats.prefillTokens : 0);
    decodeTokens.push(Number.isFinite(stats.decodeTokens) ? stats.decodeTokens : tokenIds.length);
    denoiseSteps.push(Number.isFinite(stats.denoiseSteps) ? stats.denoiseSteps : 0);
    if (Number.isFinite(stats.tokensPerForward)) {
      tokensPerForward.push(stats.tokensPerForward);
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
      name: 'diffusion_gemma',
      passed: totalMs.length > 0 && decodeTokens.some((value) => value > 0),
      duration: totalMs.reduce((sum, value) => sum + value, 0),
      error: totalMs.length === 0
        ? 'No DiffusionGemma runs completed'
        : (decodeTokens.some((value) => value > 0) ? undefined : 'No DiffusionGemma tokens generated'),
    },
  ];

  const summary = buildSuiteSummary('diffusion', results, startTime);
  const cpuStats = {
    totalMs: computeSampleStats(totalMs),
    prefillMs: computeSampleStats(prefillMs),
    denoiseMs: computeSampleStats(denoiseMs),
    vaeMs: computeSampleStats(vaeMs),
  };
  const gpuStats = { available: false };
  const avgPrefillTokens = averageRounded(prefillTokens);
  const avgDecodeTokens = averageRounded(decodeTokens);
  const avgDenoiseSteps = averageRounded(denoiseSteps);
  const basePerformanceArtifact = buildDiffusionPerformanceArtifact({
    warmupRuns,
    timedRuns,
    width: null,
    height: null,
    steps: avgDenoiseSteps,
    guidanceScale: null,
    avgPrefillTokens,
    avgDecodeTokens,
    cpuStats,
    gpuStats,
    modality: 'text',
  });
  const diffusionPerformanceArtifact = {
    ...basePerformanceArtifact,
    diffusionGemma: {
      canvasLength: runConfig.canvasLength,
      maxDenoisingSteps: runConfig.maxDenoisingSteps,
      maxNewTokens: runConfig.maxNewTokens,
      tokensPerForward: computeSampleStats(tokensPerForward),
      denoiseSteps: computeSampleStats(denoiseSteps),
    },
  };
  const prefillMsMedian = safeStatsValue(cpuStats.prefillMs?.median);
  const denoiseMsMedian = safeStatsValue(cpuStats.denoiseMs?.median);
  const totalMsMedian = safeStatsValue(cpuStats.totalMs?.median);
  const timing = buildCanonicalTiming({
    modelLoadMs: harness.modelLoadMs ?? 0,
    firstTokenMs: null,
    firstResponseMs: null,
    prefillMs: prefillMsMedian,
    decodeMs: denoiseMsMedian,
    totalRunMs: totalMsMedian,
    prefillTokensPerSec: diffusionPerformanceArtifact.throughput.prefillTokensPerSec,
    decodeTokensPerSec: diffusionPerformanceArtifact.throughput.decodeTokensPerSec,
    cacheMode,
    loadMode,
  });
  const timingDiagnostics = buildTimingDiagnostics(timing, {
    source: 'doppler',
    prefillSemantics: 'diffusion_gemma_encoder_prefill_phase',
    decodeSemantics: 'diffusion_gemma_canvas_denoising_phase',
  });
  const metricsWithContracts = buildSuiteContractMetrics(
    'diffusion',
    {
      warmupRuns,
      timedRuns,
      prompt,
      width: null,
      height: null,
      steps: avgDenoiseSteps,
      guidanceScale: null,
      maxNewTokens: runConfig.maxNewTokens,
      canvasLength: runConfig.canvasLength,
      avgPrefillTokens,
      avgDecodeTokens,
      tokensPerForward: computeSampleStats(tokensPerForward),
      latency: {
        totalMs: cpuStats.totalMs,
        prefillMs: cpuStats.prefillMs,
        denoiseMs: cpuStats.denoiseMs,
        vaeMs: cpuStats.vaeMs,
      },
      throughput: {
        prefillTokensPerSec: diffusionPerformanceArtifact.throughput.prefillTokensPerSec,
        decodeTokensPerSec: diffusionPerformanceArtifact.throughput.decodeTokensPerSec,
        decodeStepsPerSec: diffusionPerformanceArtifact.throughput.decodeStepsPerSec,
        tokensPerForward: diffusionPerformanceArtifact.diffusionGemma.tokensPerForward,
      },
      cpu: cpuStats,
      gpu: gpuStats,
      performanceArtifact: diffusionPerformanceArtifact,
    },
    harness.manifest
  );

  return {
    ...summary,
    modelId: options.modelId || harness.manifest?.modelId || 'unknown',
    cacheMode,
    loadMode,
    env: {
      library: 'doppler',
      runtime: 'browser',
      device: 'webgpu',
      browserUserAgent: typeof navigator !== 'undefined' ? (navigator.userAgent || null) : null,
      browserPlatform: typeof navigator !== 'undefined' ? (navigator.platform || null) : null,
      browserLanguage: typeof navigator !== 'undefined' ? (navigator.language || null) : null,
      browserVendor: typeof navigator !== 'undefined' ? (navigator.vendor || null) : null,
    },
    timing,
    timingDiagnostics,
    output,
    metrics: metricsWithContracts,
    memoryStats,
    deviceInfo: resolveDeviceInfo(),
    pipeline: options.keepPipeline ? harness.pipeline : null,
  };
}

export async function runDiffusionSuite(options = {}) {
  const startTime = performance.now();
  const runtimeConfig = getRuntimeConfig();
  const captureOutput = options.captureOutput === true;
  const cacheMode = normalizeCacheMode(options.cacheMode);
  const loadMode = normalizeLoadMode(options.loadMode, !!options.modelUrl, options.modelUrl);
  const benchConfig = runtimeConfig.shared?.benchmark?.run || {};
  const warmupRuns = Math.max(0, Math.floor(benchConfig.warmupRuns ?? 0));
  const timedRuns = Math.max(1, Math.floor(benchConfig.timedRuns ?? 1));
  const harness = await initializeSuiteModel(options);

  if (isDiffusionGemmaManifest(harness.manifest)) {
    return runDiffusionGemmaSuite(options, {
      startTime,
      runtimeConfig,
      captureOutput,
      cacheMode,
      loadMode,
      warmupRuns,
      timedRuns,
      harness,
    });
  }

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
  const prefillMsMedian = safeStatsValue(cpuStats.prefillMs?.median);
  const denoiseMsMedian = safeStatsValue(cpuStats.denoiseMs?.median);
  const totalMsMedian = safeStatsValue(cpuStats.totalMs?.median);
  const diffusionPerformanceArtifact = buildDiffusionPerformanceArtifact({
    warmupRuns,
    timedRuns,
    width,
    height,
    steps,
    guidanceScale,
    avgPrefillTokens,
    avgDecodeTokens,
    cpuStats,
    gpuStats,
  });
  const timing = buildCanonicalTiming({
    modelLoadMs: 0,
    firstTokenMs: null,
    firstResponseMs: null,
    prefillMs: prefillMsMedian,
    decodeMs: denoiseMsMedian,
    totalRunMs: totalMsMedian,
    prefillTokensPerSec: diffusionPerformanceArtifact.throughput.prefillTokensPerSec,
    decodeTokensPerSec: diffusionPerformanceArtifact.throughput.decodeTokensPerSec,
    cacheMode,
    loadMode,
  });
  const timingDiagnostics = buildTimingDiagnostics(timing, {
    source: 'doppler',
    prefillSemantics: 'internal_prefill_phase',
  });
  const metricsWithContracts = buildSuiteContractMetrics(
    'diffusion',
    {
      warmupRuns,
      timedRuns,
      width,
      height,
      steps,
      guidanceScale,
      prompt,
      avgPrefillTokens,
      avgDecodeTokens,
      latency: {
        totalMs: cpuStats.totalMs,
        prefillMs: cpuStats.prefillMs,
        denoiseMs: cpuStats.denoiseMs,
        vaeMs: cpuStats.vaeMs,
      },
      throughput: {
        prefillTokensPerSec: diffusionPerformanceArtifact.throughput.prefillTokensPerSec,
        decodeTokensPerSec: diffusionPerformanceArtifact.throughput.decodeTokensPerSec,
        decodeStepsPerSec: diffusionPerformanceArtifact.throughput.decodeStepsPerSec,
      },
      cpu: cpuStats,
      gpu: gpuStats,
      performanceArtifact: diffusionPerformanceArtifact,
    },
    harness.manifest
  );

  return {
    ...summary,
    modelId: options.modelId || harness.manifest?.modelId || 'unknown',
    cacheMode,
    loadMode,
    env: {
      library: 'doppler',
      runtime: 'browser',
      device: 'webgpu',
      browserUserAgent: typeof navigator !== 'undefined' ? (navigator.userAgent || null) : null,
      browserPlatform: typeof navigator !== 'undefined' ? (navigator.platform || null) : null,
      browserLanguage: typeof navigator !== 'undefined' ? (navigator.language || null) : null,
      browserVendor: typeof navigator !== 'undefined' ? (navigator.vendor || null) : null,
    },
    timing,
    timingDiagnostics,
    output,
    metrics: metricsWithContracts,
    memoryStats,
    deviceInfo: resolveDeviceInfo(),
    pipeline: options.keepPipeline ? harness.pipeline : null,
  };
}

export async function runEnergySuite(options = {}) {
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
    deviceInfo: resolveDeviceInfo(),
    pipeline: options.keepPipeline ? harness.pipeline : null,
  };
}
