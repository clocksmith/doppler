
import { initializeInference } from './test-harness.js';
import { saveReport } from '../storage/reports.js';
import { getRuntimeConfig, setRuntimeConfig } from '../config/runtime.js';
import { clearLogHistory, getDebugSnapshot } from '../debug/history.js';
import { computeSampleStats } from '../debug/stats.js';
import {
  setActiveKernelPath,
  getActiveKernelPath,
  getActiveKernelPathSource,
  getActiveKernelPathPolicy,
} from '../config/kernel-path-loader.js';
import { validateTrainingMetricsReport } from '../config/schema/training-metrics.schema.js';
import {
  resolveReportTimestamp,
  resolveRuntime,
  cloneRuntimeConfig,
  runWithRuntimeIsolationForSuite,
  sanitizeReportOutput,
  loadRuntimeConfigFromUrl,
  applyRuntimeConfigFromUrl,
  loadRuntimeProfile,
  applyRuntimeProfile,
  applyRuntimeForRun,
  normalizeManifest,
  mergeRunDefaults,
  summarizeManifestRuns,
} from './browser-harness-runtime-helpers.js';
import {
  buildSuiteSummary,
  normalizeCacheMode,
  normalizeLoadMode,
  normalizeWorkloadType,
  assertDiffusionPerformanceArtifact,
  toTimingNumber,
  buildFirstLoadComposition,
  safeToFixed,
  sampleTimingNumber,
  buildCanonicalTiming,
  buildTimingDiagnostics,
} from './browser-harness-suite-helpers.js';
import {
  resolveDeviceInfo,
  resolveKernelPathForModel,
  initializeSuiteModel,
} from './browser-harness-model-helpers.js';
import {
  resolveBenchmarkRunSettings,
  runEmbeddingSemanticChecks,
  isCoherentOutput,
  runTextInference,
  runEmbedding,
} from './browser-harness-text-helpers.js';
import { buildSuiteContractMetrics } from './browser-harness-contract-helpers.js';
import {
  runDiffusionSuite,
  runEnergySuite,
} from './browser-harness-diffusion-energy-suites.js';
import { collectTrainingArtifactsFromSuiteResult } from './browser-harness-report-helpers.js';

const TRAINING_SUITE_MODULE_PATH = '../experimental/training/suite.js';
let trainingSuiteModulePromise = null;

async function loadTrainingSuiteModule() {
  if (!trainingSuiteModulePromise) {
    trainingSuiteModulePromise = import(TRAINING_SUITE_MODULE_PATH);
  }
  return trainingSuiteModulePromise;
}

export async function runTrainingSuite(options = {}) {
  const module = await loadTrainingSuiteModule();
  return module.runTrainingSuite(options);
}

export {
  loadRuntimeConfigFromUrl,
  applyRuntimeConfigFromUrl,
  loadRuntimeProfile,
  applyRuntimeProfile,
  applyRuntimeForRun,
  buildSuiteSummary,
};

async function runTrainingBenchSuite(options = {}) {
  const module = await loadTrainingSuiteModule();
  return module.runTrainingBenchSuite(options);
}

const BROWSER_WORKLOAD_SET = Object.freeze([
  'kernels',
  'inference',
  'embedding',
  'training',
  'diffusion',
  'energy',
]);

const BROWSER_MODE_SET = Object.freeze(['verify', 'debug', 'bench', 'diagnose']);

const BROWSER_WORKLOAD_DISPATCH_MAP = Object.freeze({
  verify: Object.freeze({
    kernels: 'runKernelSuite',
    inference: 'runInferenceSuite',
    embedding: 'runEmbeddingSuite',
    training: 'runTrainingSuite',
    diffusion: 'runDiffusionSuite',
    energy: 'runEnergySuite',
  }),
  debug: Object.freeze({
    inference: 'runInferenceSuite(debug)',
    embedding: 'runEmbeddingSuite(debug)',
  }),
  diagnose: Object.freeze({
    inference: 'runInferenceSuite(diagnose)',
  }),
  bench: Object.freeze({
    inference: 'runBenchSuite',
    embedding: 'runBenchSuite',
    training: 'runBenchSuite(training)',
    diffusion: 'runBenchSuite(diffusion)',
  }),
});

export function getBrowserSupportedSuites() {
  return [...BROWSER_WORKLOAD_SET];
}

export function getBrowserSuiteDispatchMap() {
  return {
    verify: { ...BROWSER_WORKLOAD_DISPATCH_MAP.verify },
    debug: { ...BROWSER_WORKLOAD_DISPATCH_MAP.debug },
    bench: { ...BROWSER_WORKLOAD_DISPATCH_MAP.bench },
  };
}

function getAllowedWorkloadsForMode(mode) {
  return Object.keys(BROWSER_WORKLOAD_DISPATCH_MAP[mode] || {});
}

function createUnsupportedWorkloadError(requestedWorkload, context = {}) {
  const command = typeof context.command === 'string' && context.command.trim()
    ? context.command.trim()
    : 'run-browser-suite';
  const surface = typeof context.surface === 'string' && context.surface.trim()
    ? context.surface.trim()
    : 'browser';
  const mode = typeof context.mode === 'string' && context.mode.trim()
    ? context.mode.trim()
    : 'verify';
  const allowedWorkloads = getAllowedWorkloadsForMode(mode);
  const error = new Error(
    `Unsupported workload "${requestedWorkload}". Allowed workloads: ${allowedWorkloads.join(', ')}. ` +
    `command="${command}" mode="${mode}" surface="${surface}".`
  );
  error.code = 'unsupported_workload';
  error.requestedWorkload = requestedWorkload;
  error.allowedWorkloads = allowedWorkloads;
  error.command = command;
  error.mode = mode;
  error.surface = surface;
  error.details = {
    requestedWorkload,
    allowedWorkloads,
    command,
    mode,
    surface,
  };
  return error;
}

function resolveHarnessContext(options = {}) {
  const command = typeof options.command === 'string' ? options.command : null;
  const surface = typeof options.surface === 'string' ? options.surface : null;
  const mode = typeof options.mode === 'string' ? options.mode : null;
  return {
    command: command ?? 'run-browser-suite',
    mode: mode ?? command ?? 'verify',
    surface: surface ?? 'browser',
  };
}

function normalizeLegacySuite(value) {
  const suite = String(value || '').trim().toLowerCase();
  if (!suite) {
    return null;
  }
  return suite === 'benchmark' ? 'bench' : suite;
}

function normalizeMode(value, context = {}) {
  const mode = String(value || '').trim().toLowerCase();
  if (!mode) {
    return 'verify';
  }
  if (!BROWSER_MODE_SET.includes(mode)) {
    throw new Error(`Unsupported browser harness mode "${mode}" for command "${context.command || 'run-browser-suite'}".`);
  }
  return mode;
}

function resolveHarnessMode(options = {}, context = {}) {
  const explicitMode = options.mode ?? options.command ?? null;
  if (explicitMode) {
    return normalizeMode(explicitMode, context);
  }
  const legacySuite = normalizeLegacySuite(options.suite);
  if (legacySuite === 'debug' || legacySuite === 'bench') {
    return legacySuite;
  }
  return 'verify';
}

function normalizeWorkload(value, mode, context = {}) {
  const workload = String(value || '').trim().toLowerCase();
  if (!workload) {
    throw createUnsupportedWorkloadError(workload, { ...context, mode });
  }
  if (!getAllowedWorkloadsForMode(mode).includes(workload)) {
    throw createUnsupportedWorkloadError(workload, { ...context, mode });
  }
  return workload;
}

function resolveWorkload(options = {}, mode, context = {}) {
  if (options.workload) {
    return normalizeWorkload(options.workload, mode, context);
  }
  const legacySuite = normalizeLegacySuite(options.suite);
  if (legacySuite && legacySuite !== 'debug' && legacySuite !== 'bench') {
    return normalizeWorkload(legacySuite, mode, context);
  }
  if (mode === 'debug' || mode === 'bench') {
    return 'inference';
  }
  return normalizeWorkload('', mode, context);
}

function resolveDispatchSuite(mode, workload) {
  if (mode === 'debug') {
    return 'debug';
  }
  if (mode === 'bench') {
    return 'bench';
  }
  return workload;
}

async function runKernelSuite(options = {}) {
  const startTime = performance.now();
  const { testHarness, initGPU } = await import('../../tests/kernels/browser/test-page.js');
  const { runKernelSuite: runAllKernelTests } = await import('../../tests/kernels/browser/kernel-suite.js');
  await initGPU();

  const previousKernelPath = getActiveKernelPath();
  const previousKernelSource = getActiveKernelPathSource();
  const previousKernelPathPolicy = getActiveKernelPathPolicy();
  if (options.modelId) {
    await resolveKernelPathForModel(options);
  }
  let results = [];
  try {
    results = await runAllKernelTests(testHarness);
  } finally {
    setActiveKernelPath(previousKernelPath, previousKernelSource, previousKernelPathPolicy);
  }

  const summary = buildSuiteSummary('kernels', results, startTime);
  return {
    ...summary,
    deviceInfo: resolveDeviceInfo(),
  };
}

async function runInferenceSuite(options = {}) {
  const startTime = performance.now();
  const cacheMode = normalizeCacheMode(options.cacheMode);
  const loadMode = normalizeLoadMode(options.loadMode, !!options.modelUrl, options.modelUrl);
  const harness = await withHarnessPhase(
    'inference.initializeSuiteModel',
    {
      modelId: options.modelId ?? null,
      loadMode,
      cacheMode,
    },
    () => initializeSuiteModel(options)
  );
  const runtimeConfig = getRuntimeConfig();
  const modelType = harness.manifest?.modelType || 'transformer';
  if (options.expectedModelType === 'embedding' && modelType !== 'embedding') {
    throw new Error(
      `Expected an embedding model for workload "${options.workload || 'inference'}", got "${modelType}".`
    );
  }
  const safeModelLoadMs = toTimingNumber(harness.modelLoadMs, 0);

  let results;
  let output = null;
  let metrics;

  if (modelType === 'embedding') {
    const run = await runEmbedding(harness.pipeline, runtimeConfig);
    const semantic = await runEmbeddingSemanticChecks(harness.pipeline, options);
    const isValidEmbedding = run.embeddingDim > 0 && run.nonFiniteCount === 0;
    const isSemanticValid = semantic.passed;
    output = {
      mode: 'embedding',
      tokens: run.tokenCount,
      embeddingDim: run.embeddingDim,
      finiteValues: run.finiteCount,
      nonFiniteValues: run.nonFiniteCount,
      finiteRatio: Number((run.finiteRatio ?? 0).toFixed(6)),
      min: run.min == null ? null : Number(run.min.toFixed(6)),
      max: run.max == null ? null : Number(run.max.toFixed(6)),
      maxAbs: run.maxAbs == null ? null : Number(run.maxAbs.toFixed(6)),
      mean: run.mean == null ? null : Number(run.mean.toFixed(6)),
      stdDev: run.stdDev == null ? null : Number(run.stdDev.toFixed(6)),
      l2Norm: run.l2Norm == null ? null : Number(run.l2Norm.toFixed(6)),
      preview: run.preview,
      semantic: {
        passed: isSemanticValid,
        style: semantic.style,
        retrievalTop1Acc: Number(semantic.retrievalTop1Acc.toFixed(4)),
        pairAcc: Number(semantic.pairAcc.toFixed(4)),
        failedCaseIds: semantic.failedCaseIds,
        details: {
          retrieval: semantic.retrieval,
          pairs: semantic.pairs,
        },
      },
    };
    results = [
      {
        name: 'embedding',
        passed: isValidEmbedding,
        duration: run.durationMs,
        error: isValidEmbedding
          ? undefined
          : (
            run.embeddingDim <= 0
              ? 'No embedding returned'
              : `Embedding contains non-finite values (${run.nonFiniteCount}/${run.embeddingDim})`
          ),
      },
      {
        name: 'embedding-semantic',
        passed: isSemanticValid,
        duration: semantic.durationMs,
        error: isSemanticValid
          ? undefined
          : (
            `Semantic checks below threshold: retrieval=${(semantic.retrievalTop1Acc * 100).toFixed(1)}% `
            + `(min ${(semantic.minRetrievalTop1Acc * 100).toFixed(1)}%), `
            + `pairs=${(semantic.pairAcc * 100).toFixed(1)}% `
            + `(min ${(semantic.minPairAcc * 100).toFixed(1)}%). `
            + (semantic.failedCaseIds.length > 0 ? `Failed: ${semantic.failedCaseIds.join(', ')}` : '')
          ),
      },
    ];
    metrics = {
      prompt: run.prompt,
      embeddingTokens: run.tokenCount,
      embeddingDim: run.embeddingDim,
      finiteValues: run.finiteCount,
      finiteRatio: Number((run.finiteRatio ?? 0).toFixed(6)),
      nonFiniteValues: run.nonFiniteCount,
      embeddingMin: run.min == null ? null : Number(run.min.toFixed(6)),
      embeddingMax: run.max == null ? null : Number(run.max.toFixed(6)),
      embeddingMaxAbs: run.maxAbs == null ? null : Number(run.maxAbs.toFixed(6)),
      embeddingMean: run.mean == null ? null : Number(run.mean.toFixed(6)),
      embeddingStdDev: run.stdDev == null ? null : Number(run.stdDev.toFixed(6)),
      embeddingL2Norm: run.l2Norm == null ? null : Number(run.l2Norm.toFixed(6)),
      embeddingMs: Number(run.durationMs.toFixed(2)),
      semanticPassed: isSemanticValid,
      semanticDurationMs: Number(semantic.durationMs.toFixed(2)),
      semanticRetrievalTop1Acc: Number(semantic.retrievalTop1Acc.toFixed(4)),
      semanticPairAcc: Number(semantic.pairAcc.toFixed(4)),
      semanticRetrievalPassed: semantic.retrievalPassed,
      semanticRetrievalTotal: semantic.retrievalTotal,
      semanticPairPassed: semantic.pairPassed,
      semanticPairTotal: semantic.pairTotal,
      semanticMinRetrievalTop1Acc: Number(semantic.minRetrievalTop1Acc.toFixed(4)),
      semanticMinPairAcc: Number(semantic.minPairAcc.toFixed(4)),
      semanticPairMarginThreshold: Number(semantic.pairMarginThreshold.toFixed(4)),
      semanticStyle: semantic.style,
      semanticFailedCases: semantic.failedCaseIds,
      semanticDetails: {
        retrieval: semantic.retrieval,
        pairs: semantic.pairs,
      },
      modelLoadMs: safeModelLoadMs,
      endToEndMs: safeToFixed(safeModelLoadMs + run.durationMs),
      embeddingPreview: run.preview,
    };
  } else {
    const run = await runTextInference(
      harness.pipeline,
      runtimeConfig,
      options.inferenceInput ?? null
    );
    const coherent = isCoherentOutput(run.tokens, run.output);
    results = [
      {
        name: 'generation',
        passed: run.tokens.length > 0 && coherent,
        duration: run.durationMs,
        error: run.tokens.length === 0
          ? 'No tokens generated'
          : (!coherent ? 'Output dominated by padding or special tokens' : undefined),
      },
    ];
    output = run.output;
    metrics = {
      prompt: run.prompt,
      maxTokens: run.maxTokens,
      tokensGenerated: run.tokens.length,
      tokensPerSec: safeToFixed(run.tokensPerSec),
      totalRunMs: safeToFixed(run.phase.totalMs),
      firstTokenMs: safeToFixed(run.phase.ttftMs),
      firstResponseMs: safeToFixed(safeModelLoadMs + run.phase.ttftMs),
      prefillMs: safeToFixed(run.phase.prefillMs),
      decodeMs: safeToFixed(run.phase.decodeMs),
      prefillTokens: Math.round(run.phase.prefillTokens),
      decodeTokens: Math.round(run.phase.decodeTokens),
      prefillTokensPerSec: safeToFixed(run.phase.prefillTokensPerSec),
      prefillTokensPerSecTtft: safeToFixed(run.phase.prefillTokensPerSecTtft),
      decodeTokensPerSec: safeToFixed(run.phase.decodeTokensPerSec),
      modelLoadMs: safeModelLoadMs,
      gpu: run.phase.gpu,
      plePreparedTokenCache: run.phase.plePreparedTokenCache ?? null,
      prefillProfileSteps: run.phase.prefillProfileSteps,
      decodeProfileSteps: run.phase.decodeProfileSteps,
      executionPlan: run.phase.executionPlan,
      kernelPathId: run.phase.kernelPathId,
      kernelPathSource: run.phase.kernelPathSource,
      generationDiagnostics: run.tokenDiagnostics,
      operatorDiagnostics: run.phase.operatorDiagnostics ?? null,
    };
  }

  const memoryStats = typeof harness.pipeline?.getMemoryStats === 'function'
    ? harness.pipeline.getMemoryStats()
    : null;
  if (typeof harness.pipeline.unload === 'function' && !options.keepPipeline) {
    await harness.pipeline.unload();
  }

  const summary = buildSuiteSummary(options.suiteName || 'inference', results, startTime);
  const timing = buildCanonicalTiming({
    modelLoadMs: safeModelLoadMs,
    firstTokenMs: metrics.firstTokenMs ?? null,
    firstResponseMs: Number.isFinite(metrics.firstTokenMs)
      ? safeModelLoadMs + metrics.firstTokenMs
      : null,
    prefillMs: metrics.prefillMs ?? 0,
    decodeMs: metrics.decodeMs ?? 0,
    decodeMsPerTokenP50: metrics.decodeMsPerTokenP50 ?? null,
    decodeMsPerTokenP95: metrics.decodeMsPerTokenP95 ?? null,
    decodeMsPerTokenP99: metrics.decodeMsPerTokenP99 ?? null,
    totalRunMs: metrics.totalRunMs ?? metrics.decodeMs ?? 0,
    decodeTokensPerSec: metrics.decodeTokensPerSec,
    prefillTokensPerSec: metrics.prefillTokensPerSec,
    cacheMode,
    loadMode,
  });
  const timingDiagnostics = buildTimingDiagnostics(timing, {
    source: 'doppler',
    prefillSemantics: 'internal_prefill_phase',
  });
  const firstLoad = buildFirstLoadComposition({
    modelLoadMs: timing.modelLoadMs,
    firstTokenMs: timing.firstTokenMs,
    firstResponseMs: timing.firstResponseMs,
  });
  const metricsWithContracts = buildSuiteContractMetrics(
    options.suiteName || 'inference',
    metrics,
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
    firstLoad,
    output,
    metrics: metricsWithContracts,
    memoryStats,
    deviceInfo: resolveDeviceInfo(),
    pipeline: options.keepPipeline ? harness.pipeline : null,
  };
}

async function runBenchSuite(options = {}) {
  const startTime = performance.now();
  const runtimeConfig = getRuntimeConfig();
  const defaultBenchRun = resolveBenchmarkRunSettings(runtimeConfig);
  const warmupRuns = defaultBenchRun.warmupRuns;
  const timedRuns = defaultBenchRun.timedRuns;
  const cacheMode = normalizeCacheMode(options.cacheMode);
  const loadMode = normalizeLoadMode(options.loadMode, !!options.modelUrl, options.modelUrl);
  const workloadType = normalizeWorkloadType(
    options.workloadType
    ?? (
      options.mode === 'bench' && (options.workload === 'training' || options.workload === 'diffusion')
        ? options.workload
        : null
    )
  );

  if (workloadType === 'training') {
    const trainingBench = await runTrainingBenchSuite({
      ...options,
      benchRun: defaultBenchRun,
      workloadType,
    });
    const trainingReport = trainingBench?.metrics?.trainingMetricsReport;
    if (Array.isArray(trainingReport) && trainingReport.length > 0) {
      validateTrainingMetricsReport(trainingReport);
    }
    const runStats = trainingBench?.metrics?.latency?.runMs || computeSampleStats([]);
    const stepStats = trainingBench?.metrics?.latency?.stepMs || computeSampleStats([]);
    const throughputStats = trainingBench?.metrics?.throughput?.stepsPerSec || computeSampleStats([]);
    const timing = buildCanonicalTiming({
      modelLoadMs: 0,
      firstTokenMs: null,
      firstResponseMs: null,
      prefillMs: null,
      decodeMs: stepStats.median,
      totalRunMs: runStats.median,
      decodeTokensPerSec: throughputStats.median,
      prefillTokensPerSec: null,
      cacheMode,
      loadMode,
    });
    const timingDiagnostics = buildTimingDiagnostics(timing, {
      source: 'doppler',
      prefillSemantics: 'not_applicable_training_workload',
    });
    const firstLoad = buildFirstLoadComposition({
      modelLoadMs: timing.modelLoadMs,
      firstTokenMs: timing.firstTokenMs,
      firstResponseMs: timing.firstResponseMs,
    });
    return {
      ...trainingBench,
      modelId: trainingBench.modelId || options.modelId || options.modelUrl || 'training',
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
      firstLoad,
      output: null,
      memoryStats: null,
      deviceInfo: trainingBench.deviceInfo ?? resolveDeviceInfo(),
      pipeline: null,
    };
  }

  if (workloadType === 'diffusion') {
    const diffusionBench = await runDiffusionSuite({
      ...options,
      command: 'bench',
      workload: 'diffusion',
      captureOutput: options.captureOutput === true,
      cacheMode,
      loadMode,
    });

    const benchResults = [
      {
        name: 'benchmark-diffusion',
        passed: diffusionBench.passed > 0 && diffusionBench.failed === 0,
        duration: diffusionBench.duration,
        error: diffusionBench.failed === 0 ? undefined : 'Diffusion benchmark run failed.',
      },
    ];
    const summary = buildSuiteSummary('bench', benchResults, startTime);

    return {
      ...diffusionBench,
      ...summary,
      suite: 'bench',
      results: benchResults,
      metrics: {
        ...(diffusionBench.metrics || {}),
        workloadType: 'diffusion',
      },
    };
  }

  const harness = await withHarnessPhase(
    'bench.initializeSuiteModel',
    {
      modelId: options.modelId ?? null,
      loadMode,
      cacheMode,
    },
    () => initializeSuiteModel(options)
  );
  const benchRun = resolveBenchmarkRunSettings(runtimeConfig, harness.pipeline ?? harness);
  const modelType = harness.manifest?.modelType || 'transformer';
  if (options.expectedModelType === 'embedding' && modelType !== 'embedding') {
    throw new Error(
      `Expected an embedding model for bench workload "${options.workload || 'inference'}", got "${modelType}".`
    );
  }
  const safeModelLoadMs = toTimingNumber(harness.modelLoadMs, 0);

  let results;
  let metrics;
  let output = null;
  let timing;

  if (modelType === 'embedding') {
    const durations = [];
    const timedDurations = [];
    const embeddingDims = [];
    const embeddingTokenCounts = [];
    const embeddingNorms = [];
    let firstTimedEmbeddingMs = null;
    let invalidRuns = 0;
    let totalNonFiniteValues = 0;
    for (let i = 0; i < warmupRuns + timedRuns; i++) {
      harness.pipeline.reset?.();
      const run = await runEmbedding(harness.pipeline, runtimeConfig, benchRun);
      if (i >= warmupRuns) {
        timedDurations.push(run.durationMs);
        if (firstTimedEmbeddingMs == null) {
          firstTimedEmbeddingMs = run.durationMs;
        }
        totalNonFiniteValues += run.nonFiniteCount;
        if (Number.isFinite(run.tokenCount)) {
          embeddingTokenCounts.push(run.tokenCount);
        }
        if (Number.isFinite(run.l2Norm)) {
          embeddingNorms.push(run.l2Norm);
        }
        if (run.embeddingDim > 0 && run.nonFiniteCount === 0) {
          durations.push(run.durationMs);
          embeddingDims.push(run.embeddingDim);
        } else {
          invalidRuns++;
        }
      }
    }

    const embeddingMsStats = computeSampleStats(durations);
    const timedEmbeddingMsStats = computeSampleStats(timedDurations);
    const embeddingDimStats = computeSampleStats(embeddingDims);
    const embeddingTokensStats = computeSampleStats(embeddingTokenCounts);
    const embeddingNormStats = computeSampleStats(embeddingNorms);
    const avgMs = embeddingMsStats.mean;

    results = [
      {
        name: 'benchmark-embedding',
        passed: durations.length > 0 && invalidRuns === 0,
        duration: durations.reduce((sum, value) => sum + value, 0),
        error: durations.length > 0
          ? (
            invalidRuns === 0
              ? undefined
              : `Invalid embedding runs: ${invalidRuns} (non-finite values observed)`
          )
          : 'No valid embedding benchmark runs completed',
      },
    ];

    metrics = {
      warmupRuns,
      timedRuns,
      validRuns: durations.length,
      invalidRuns,
      invalidRatePct: Number((timedRuns > 0 ? (invalidRuns / timedRuns) * 100 : 0).toFixed(2)),
      prompt: benchRun.promptLabel,
      embeddingDim: Math.round(embeddingDims.reduce((a, b) => a + b, 0) / (embeddingDims.length || 1)),
      nonFiniteValues: totalNonFiniteValues,
      firstTimedEmbeddingMs: Number((firstTimedEmbeddingMs ?? 0).toFixed(2)),
      minEmbeddingMs: Number(embeddingMsStats.min.toFixed(2)),
      medianEmbeddingMs: Number(embeddingMsStats.median.toFixed(2)),
      p95EmbeddingMs: Number(embeddingMsStats.p95.toFixed(2)),
      p99EmbeddingMs: Number(embeddingMsStats.p99.toFixed(2)),
      maxEmbeddingMs: Number(embeddingMsStats.max.toFixed(2)),
      stdDevEmbeddingMs: Number(embeddingMsStats.stdDev.toFixed(2)),
      ci95EmbeddingMs: Number(embeddingMsStats.ci95.toFixed(2)),
      avgEmbeddingMs: Number(avgMs.toFixed(2)),
      avgEmbeddingsPerSec: Number((avgMs > 0 ? (1000 / avgMs) : 0).toFixed(2)),
      avgEmbeddingTokens: Number(embeddingTokensStats.mean.toFixed(2)),
      avgEmbeddingL2Norm: Number(embeddingNormStats.mean.toFixed(4)),
      modelLoadMs: safeModelLoadMs,
      latency: {
        timedEmbeddingMs: timedEmbeddingMsStats,
        embeddingMs: embeddingMsStats,
      },
      dimensions: {
        embedding: embeddingDimStats,
      },
      embedding: {
        tokens: embeddingTokensStats,
        l2Norm: embeddingNormStats,
      },
    };

    const timedStats = computeSampleStats(durations);
    timing = buildCanonicalTiming({
      modelLoadMs: safeModelLoadMs,
      firstTokenMs: null,
      firstResponseMs: Number.isFinite(firstTimedEmbeddingMs)
        ? safeModelLoadMs + firstTimedEmbeddingMs
        : null,
      prefillMs: null,
      decodeMs: null,
      totalRunMs: timedStats.median,
      cacheMode,
      loadMode,
    });
  } else {
    const tokensPerSec = [];
    const durations = [];
    const tokensGenerated = [];
    const decodeMsPerToken = [];
    const ttftMs = [];
    const prefillMs = [];
    const decodeMs = [];
    const prefillTokens = [];
    const decodeTokens = [];
    const decodeTokensPerSec = [];
    const prefillTokensPerSec = [];
    const prefillTokensPerSecTtft = [];
    const gpuPrefillMs = [];
    const gpuDecodeMs = [];
    const gpuDecodeRecordMs = [];
    const gpuDecodeSubmitWaitMs = [];
    const gpuDecodeReadbackWaitMs = [];
    const gpuDecodeOrchestrationMs = [];
    const gpuPrefillRecordMs = [];
    const gpuPrefillSubmitWaitMs = [];
    const singleTokenSubmitWaitMs = [];
    const singleTokenReadbackWaitMs = [];
    const singleTokenOrchestrationMs = [];
    const batchedForwardCalls = [];
    const unbatchedForwardCalls = [];
    const totalBatchedTimeMs = [];
    const totalUnbatchedTimeMs = [];
    const gpuSubmissions = [];
    const plePreparedTokenCacheHits = [];
    const plePreparedTokenCacheMisses = [];
    const plePreparedTokenCacheEntries = [];
    const plePreparedTokenCacheBytes = [];

    let generatedText = null;
    let generatedPromptInput = null;
    let lastDecodeMode = null;
    let lastBatchGuardReason = null;
    for (let i = 0; i < warmupRuns + timedRuns; i++) {
      harness.pipeline.reset?.();
      const run = await withHarnessPhase(
        `bench.runTextInference[${i}]`,
        {
          modelId: options.modelId ?? harness.manifest?.modelId ?? null,
          loadMode,
          cacheMode,
          warmupRuns,
          timedRuns,
        },
        () => runTextInference(harness.pipeline, runtimeConfig, {
          ...benchRun,
          ...(options.inferenceInput ?? {}),
        })
      );
      if (i === warmupRuns + timedRuns - 1) {
        generatedText = run?.output ?? null;
        generatedPromptInput = run?.promptInput ?? null;
        lastDecodeMode = run?.phase?.decodeMode ?? null;
        lastBatchGuardReason = run?.phase?.batchGuardReason ?? null;
      }
      if (i >= warmupRuns) {
        const phase = run?.phase ?? {};
        const phaseTokens = Array.isArray(run?.tokens) ? run.tokens : [];
        const phaseGpu = phase.gpu;
        const phaseBatching = phase.batching;
        const phasePlePreparedTokenCache = phase.plePreparedTokenCache;
        tokensPerSec.push(run?.tokensPerSec);
        durations.push(run?.durationMs);
        tokensGenerated.push(phaseTokens.length);
        ttftMs.push(phase.ttftMs);
        prefillMs.push(phase.prefillMs);
        decodeMs.push(phase.decodeMs);
        prefillTokens.push(phase.prefillTokens);
        decodeTokens.push(phase.decodeTokens);
        decodeTokensPerSec.push(phase.decodeTokensPerSec);
        prefillTokensPerSec.push(phase.prefillTokensPerSec);
        prefillTokensPerSecTtft.push(phase.prefillTokensPerSecTtft);
        if (phase.decodeMs > 0 && phase.decodeTokens > 0) {
          decodeMsPerToken.push(phase.decodeMs / phase.decodeTokens);
        }
        if (Number.isFinite(phaseGpu?.prefillMs)) gpuPrefillMs.push(phaseGpu.prefillMs);
        if (Number.isFinite(phaseGpu?.decodeMs)) gpuDecodeMs.push(phaseGpu.decodeMs);
        if (Number.isFinite(phaseGpu?.decodeRecordMs)) gpuDecodeRecordMs.push(phaseGpu.decodeRecordMs);
        if (Number.isFinite(phaseGpu?.decodeSubmitWaitMs)) gpuDecodeSubmitWaitMs.push(phaseGpu.decodeSubmitWaitMs);
        if (Number.isFinite(phaseGpu?.decodeReadbackWaitMs)) gpuDecodeReadbackWaitMs.push(phaseGpu.decodeReadbackWaitMs);
        if (Number.isFinite(phaseGpu?.prefillRecordMs)) gpuPrefillRecordMs.push(phaseGpu.prefillRecordMs);
        if (Number.isFinite(phaseGpu?.prefillSubmitWaitMs)) gpuPrefillSubmitWaitMs.push(phaseGpu.prefillSubmitWaitMs);
        if (Number.isFinite(phaseGpu?.decodeOrchestrationMs)) {
          gpuDecodeOrchestrationMs.push(phaseGpu.decodeOrchestrationMs);
        }
        if (Number.isFinite(phaseGpu?.singleTokenSubmitWaitMs)) singleTokenSubmitWaitMs.push(phaseGpu.singleTokenSubmitWaitMs);
        if (Number.isFinite(phaseGpu?.singleTokenReadbackWaitMs)) singleTokenReadbackWaitMs.push(phaseGpu.singleTokenReadbackWaitMs);
        if (Number.isFinite(phaseGpu?.singleTokenOrchestrationMs)) singleTokenOrchestrationMs.push(phaseGpu.singleTokenOrchestrationMs);
        if (Number.isFinite(phaseBatching?.batchedForwardCalls)) batchedForwardCalls.push(phaseBatching.batchedForwardCalls);
        if (Number.isFinite(phaseBatching?.unbatchedForwardCalls)) unbatchedForwardCalls.push(phaseBatching.unbatchedForwardCalls);
        if (Number.isFinite(phaseBatching?.totalBatchedTimeMs)) totalBatchedTimeMs.push(phaseBatching.totalBatchedTimeMs);
        if (Number.isFinite(phaseBatching?.totalUnbatchedTimeMs)) totalUnbatchedTimeMs.push(phaseBatching.totalUnbatchedTimeMs);
        if (Number.isFinite(phaseBatching?.gpuSubmissions)) gpuSubmissions.push(phaseBatching.gpuSubmissions);
        if (Number.isFinite(phasePlePreparedTokenCache?.hits)) plePreparedTokenCacheHits.push(phasePlePreparedTokenCache.hits);
        if (Number.isFinite(phasePlePreparedTokenCache?.misses)) plePreparedTokenCacheMisses.push(phasePlePreparedTokenCache.misses);
        if (Number.isFinite(phasePlePreparedTokenCache?.entries)) plePreparedTokenCacheEntries.push(phasePlePreparedTokenCache.entries);
        if (Number.isFinite(phasePlePreparedTokenCache?.bytes)) plePreparedTokenCacheBytes.push(phasePlePreparedTokenCache.bytes);
      }
    }

    const totalMsStats = computeSampleStats(durations);
    const tokensPerSecStats = computeSampleStats(tokensPerSec);
    const decodeTokensPerSecStats = computeSampleStats(decodeTokensPerSec);
    const prefillTokensPerSecStats = computeSampleStats(prefillTokensPerSec);
    const prefillTokensPerSecTtftStats = computeSampleStats(prefillTokensPerSecTtft);
    const decodeMsPerTokenStats = computeSampleStats(decodeMsPerToken);
    const ttftMsStats = computeSampleStats(ttftMs);
    const prefillMsStats = computeSampleStats(prefillMs);
    const decodeMsStats = computeSampleStats(decodeMs);
    const tokensGeneratedStats = computeSampleStats(tokensGenerated);
    const prefillTokensStats = computeSampleStats(prefillTokens);
    const decodeTokensStats = computeSampleStats(decodeTokens);
    const hasGpuStats = gpuPrefillMs.length > 0 || gpuDecodeMs.length > 0 || gpuDecodeRecordMs.length > 0
      || gpuDecodeSubmitWaitMs.length > 0 || gpuDecodeReadbackWaitMs.length > 0
      || gpuDecodeOrchestrationMs.length > 0
      || singleTokenSubmitWaitMs.length > 0 || singleTokenReadbackWaitMs.length > 0
      || singleTokenOrchestrationMs.length > 0;
    const gpuPhaseStats = hasGpuStats
      ? {
        prefillMs: computeSampleStats(gpuPrefillMs),
        decodeMs: computeSampleStats(gpuDecodeMs),
        decodeRecordMs: computeSampleStats(gpuDecodeRecordMs),
        decodeSubmitWaitMs: computeSampleStats(gpuDecodeSubmitWaitMs),
        decodeReadbackWaitMs: computeSampleStats(gpuDecodeReadbackWaitMs),
        decodeOrchestrationMs: computeSampleStats(gpuDecodeOrchestrationMs),
        prefillRecordMs: computeSampleStats(gpuPrefillRecordMs),
        prefillSubmitWaitMs: computeSampleStats(gpuPrefillSubmitWaitMs),
        singleTokenSubmitWaitMs: computeSampleStats(singleTokenSubmitWaitMs),
        singleTokenReadbackWaitMs: computeSampleStats(singleTokenReadbackWaitMs),
        singleTokenOrchestrationMs: computeSampleStats(singleTokenOrchestrationMs),
      }
      : null;
    const hasBatchingStats = batchedForwardCalls.length > 0
      || unbatchedForwardCalls.length > 0
      || totalBatchedTimeMs.length > 0
      || totalUnbatchedTimeMs.length > 0
      || gpuSubmissions.length > 0;
    const batchingPhaseStats = hasBatchingStats
      ? {
        batchedForwardCalls: computeSampleStats(batchedForwardCalls),
        unbatchedForwardCalls: computeSampleStats(unbatchedForwardCalls),
        totalBatchedTimeMs: computeSampleStats(totalBatchedTimeMs),
        totalUnbatchedTimeMs: computeSampleStats(totalUnbatchedTimeMs),
        gpuSubmissions: computeSampleStats(gpuSubmissions),
      }
      : null;
    const hasPlePreparedTokenCacheStats = plePreparedTokenCacheHits.length > 0
      || plePreparedTokenCacheMisses.length > 0
      || plePreparedTokenCacheEntries.length > 0
      || plePreparedTokenCacheBytes.length > 0;
    const plePreparedTokenCacheStats = hasPlePreparedTokenCacheStats
      ? {
        hits: computeSampleStats(plePreparedTokenCacheHits),
        misses: computeSampleStats(plePreparedTokenCacheMisses),
        entries: computeSampleStats(plePreparedTokenCacheEntries),
        bytes: computeSampleStats(plePreparedTokenCacheBytes),
      }
      : null;

    results = [
      {
        name: 'benchmark',
        passed: tokensPerSec.length > 0,
        duration: durations.reduce((sum, value) => sum + value, 0),
        error: tokensPerSec.length > 0 ? undefined : 'No benchmark runs completed',
      },
    ];

    const normalizedFirstTokenMs = sampleTimingNumber(ttftMsStats, 'median', null);

    metrics = {
      warmupRuns,
      timedRuns,
      ...(Number.isFinite(benchRun?.seed) ? { seed: benchRun.seed } : {}),
      prompt: benchRun.promptLabel,
      maxTokens: benchRun.maxTokens,
      decodeTokensPerSec: sampleTimingNumber(decodeTokensPerSecStats, 'median'),
      avgTokensGenerated: Math.round(tokensGeneratedStats.mean),
      avgPrefillTokens: Math.round(prefillTokensStats.mean),
      avgDecodeTokens: Math.round(decodeTokensStats.mean),
      medianPrefillTokensPerSec: sampleTimingNumber(prefillTokensPerSecStats, 'median'),
      avgPrefillTokensPerSec: sampleTimingNumber(prefillTokensPerSecStats, 'mean'),
      medianPrefillTokensPerSecTtft: sampleTimingNumber(prefillTokensPerSecTtftStats, 'median'),
      avgPrefillTokensPerSecTtft: sampleTimingNumber(prefillTokensPerSecTtftStats, 'mean'),
      avgDecodeTokensPerSec: sampleTimingNumber(decodeTokensPerSecStats, 'mean'),
      firstTokenMs: normalizedFirstTokenMs,
      firstResponseMs: safeToFixed(safeModelLoadMs + normalizedFirstTokenMs, null),
      prefillMs: sampleTimingNumber(prefillMsStats, 'median'),
      decodeMs: sampleTimingNumber(decodeMsStats, 'median'),
      totalRunMs: sampleTimingNumber(totalMsStats, 'median'),
      decodeMsPerTokenP50: sampleTimingNumber(decodeMsPerTokenStats, 'median'),
      decodeMsPerTokenP95: sampleTimingNumber(decodeMsPerTokenStats, 'p95'),
      decodeMsPerTokenP99: sampleTimingNumber(decodeMsPerTokenStats, 'p99'),
      avgPrefillMs: sampleTimingNumber(prefillMsStats, 'mean'),
      modelLoadMs: safeModelLoadMs,
      throughput: {
        tokensPerSec: tokensPerSecStats,
        prefillTokensPerSec: prefillTokensPerSecStats,
        prefillTokensPerSecTtft: prefillTokensPerSecTtftStats,
        decodeTokensPerSec: decodeTokensPerSecStats,
      },
      latency: {
        totalMs: totalMsStats,
        prefillMs: prefillMsStats,
        decodeMs: decodeMsStats,
        firstTokenMs: ttftMsStats,
      },
      tokens: {
        generated: tokensGeneratedStats,
        prefill: prefillTokensStats,
        decode: decodeTokensStats,
      },
      gpu: gpuPhaseStats,
      batching: batchingPhaseStats,
      plePreparedTokenCache: plePreparedTokenCacheStats,
      decodeMode: lastDecodeMode,
      batchGuardReason: lastBatchGuardReason,
      generatedText,
      promptInput: generatedPromptInput,
    };

    timing = buildCanonicalTiming({
      modelLoadMs: safeModelLoadMs,
      firstTokenMs: normalizedFirstTokenMs,
      firstResponseMs: Number.isFinite(normalizedFirstTokenMs)
        ? safeModelLoadMs + normalizedFirstTokenMs
        : null,
      prefillMs: prefillMsStats?.median ?? null,
      decodeMs: decodeMsStats?.median ?? null,
      decodeMsPerTokenP50: decodeMsPerTokenStats?.median ?? null,
      decodeMsPerTokenP95: decodeMsPerTokenStats?.p95 ?? null,
      decodeMsPerTokenP99: decodeMsPerTokenStats?.p99 ?? null,
      totalRunMs: totalMsStats.median,
      decodeTokensPerSec: decodeTokensPerSecStats?.median,
      prefillTokensPerSec: prefillTokensPerSecStats?.median,
      prefillTokensPerSecTtft: prefillTokensPerSecTtftStats?.median,
      cacheMode,
      loadMode,
    });
  }

  const memoryStats = typeof harness.pipeline?.getMemoryStats === 'function'
    ? harness.pipeline.getMemoryStats()
    : null;

  if (typeof harness.pipeline.unload === 'function' && !options.keepPipeline) {
    await harness.pipeline.unload();
  }

  const summary = buildSuiteSummary('bench', results, startTime);
  const timingDiagnostics = buildTimingDiagnostics(timing, {
    source: 'doppler',
    prefillSemantics: 'internal_prefill_phase',
  });
  const firstLoad = buildFirstLoadComposition({
    modelLoadMs: timing.modelLoadMs,
    firstTokenMs: timing.firstTokenMs,
    firstResponseMs: timing.firstResponseMs,
  });
  const metricsWithContracts = buildSuiteContractMetrics('bench', metrics, harness.manifest);
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
    firstLoad,
    output,
    metrics: metricsWithContracts,
    memoryStats,
    deviceInfo: resolveDeviceInfo(),
    pipeline: options.keepPipeline ? harness.pipeline : null,
  };
}

async function dispatchBrowserSuite(mode, workload, options) {
  if (mode === 'verify' && workload === 'kernels') {
    return runKernelSuite(options);
  }
  if (workload === 'embedding') {
    return runInferenceSuite({
      ...options,
      suiteName: 'embedding',
      expectedModelType: options.expectedModelType ?? 'embedding',
    });
  }
  if (mode === 'verify' && workload === 'training') {
    return runTrainingSuite(options);
  }
  if (mode === 'bench') {
    return runBenchSuite(options);
  }
  if (mode === 'verify' && workload === 'diffusion') {
    return runDiffusionSuite(options);
  }
  if (mode === 'verify' && workload === 'energy') {
    return runEnergySuite(options);
  }
  if (mode === 'debug' && workload === 'inference') {
    return runInferenceSuite({ ...options, suiteName: 'debug' });
  }
  if (mode === 'diagnose' && workload === 'inference') {
    return runInferenceSuite({ ...options, suiteName: 'diagnose' });
  }
  if (workload === 'inference') {
    return runInferenceSuite({ ...options, suiteName: 'inference' });
  }
  return null;
}

function shouldCaptureDebugSnapshot(mode, runtimeConfig) {
  const debug = runtimeConfig?.shared?.debug ?? {};
  const logLevel = String(debug.logLevel?.defaultLogLevel ?? '').toLowerCase();
  return mode === 'debug'
    || debug.trace?.enabled === true
    || debug.pipeline?.enabled === true
    || (Array.isArray(debug.probes) && debug.probes.length > 0)
    || debug.profiler?.enabled === true
    || logLevel === 'debug'
    || logLevel === 'verbose';
}

function createHarnessPhaseError(error, phase, context = {}) {
  const message = error?.message || String(error);
  const wrapped = new Error(
    `Browser harness phase "${phase}" failed: ${message}`,
    error instanceof Error ? { cause: error } : undefined
  );
  wrapped.name = error?.name || 'Error';
  if (error?.code !== undefined) {
    wrapped.code = error.code;
  }
  wrapped.details = {
    ...(error?.details && typeof error.details === 'object' ? error.details : {}),
    harnessPhase: phase,
    ...context,
  };
  return wrapped;
}

async function withHarnessPhase(phase, context, run) {
  try {
    return await run();
  } catch (error) {
    if (error?.details?.harnessPhase) {
      throw error;
    }
    throw createHarnessPhaseError(error, phase, context);
  }
}

export async function runBrowserSuite(options = {}) {
  return runWithRuntimeIsolationForSuite(async () => {
    const suiteTimestamp = resolveReportTimestamp(options.timestamp, 'runBrowserSuite timestamp');
    const harnessContext = resolveHarnessContext(options);
    const mode = resolveHarnessMode(options, harnessContext);
    const workload = resolveWorkload(options, mode, harnessContext);
    const suite = resolveDispatchSuite(mode, workload);
    const captureDebugSnapshot = shouldCaptureDebugSnapshot(mode, getRuntimeConfig());
    if (captureDebugSnapshot) {
      clearLogHistory();
    }
    const suiteResult = await withHarnessPhase(
      'dispatchBrowserSuite',
      {
        mode,
        workload,
        modelId: options.modelId ?? null,
        loadMode: options.loadMode ?? null,
      },
      () => dispatchBrowserSuite(mode, workload, {
        ...options,
        mode,
        workload,
        suite,
      })
    );
    if (!suiteResult) {
      throw createUnsupportedWorkloadError(workload, { ...harnessContext, mode });
    }
    const debugSnapshot = captureDebugSnapshot ? getDebugSnapshot() : null;

    if (mode === 'bench' && suiteResult?.metrics?.workloadType === 'training') {
      const trainingReport = suiteResult?.metrics?.trainingMetricsReport;
      if (Array.isArray(trainingReport) && trainingReport.length > 0) {
        validateTrainingMetricsReport(trainingReport);
      }
    }
    if (mode === 'verify' && workload === 'diffusion') {
      assertDiffusionPerformanceArtifact(suiteResult?.metrics, 'diffusion verify');
    }
    if (mode === 'bench' && suiteResult?.metrics?.workloadType === 'diffusion') {
      assertDiffusionPerformanceArtifact(suiteResult?.metrics, 'diffusion bench');
    }

    const modelId = suiteResult.modelId || options.modelId || options.modelUrl || workload || suite;
    const reportOutput = sanitizeReportOutput(suiteResult.output);
    const trainingArtifacts = collectTrainingArtifactsFromSuiteResult(suiteResult);
    const ulArtifacts = trainingArtifacts.ulArtifacts;
    const distillArtifacts = trainingArtifacts.distillArtifacts;
    const checkpointResumeTimeline = trainingArtifacts.checkpointResumeTimeline;
    const report = {
      mode,
      workload,
      suite,
      modelId,
      runtimeProfile: options.runtimeProfile ?? null,
      deviceInfo: suiteResult.deviceInfo ?? null,
      results: suiteResult.results,
      durationMs: suiteResult.duration,
      timestamp: suiteTimestamp,
      metrics: suiteResult.metrics ?? null,
      output: reportOutput,
      memory: suiteResult.memoryStats ?? null,
      debugSnapshot,
      ...options.report,
    };
    if (ulArtifacts.length > 0 || distillArtifacts.length > 0 || checkpointResumeTimeline.length > 0) {
      report.lineage = {
        ...(report.lineage && typeof report.lineage === 'object' ? report.lineage : {}),
        training: {
          ...(
            report.lineage?.training && typeof report.lineage.training === 'object'
              ? report.lineage.training
              : {}
          ),
          ...(ulArtifacts.length > 0 ? { ulArtifacts } : {}),
          ...(distillArtifacts.length > 0 ? { distillArtifacts } : {}),
          ...(checkpointResumeTimeline.length > 0 ? { checkpointResumeTimeline } : {}),
        },
      };
    }
    if (!report.timestamp) {
      report.timestamp = suiteTimestamp;
    }
    const reportInfo = await saveReport(modelId, report, { timestamp: report.timestamp });
    return { ...suiteResult, mode, workload, debugSnapshot, report, reportInfo };
  });
}

export async function runBrowserManifest(manifest, options = {}) {
  const normalized = normalizeManifest(manifest);
  const results = [];
  const manifestTimestamp = resolveReportTimestamp(options.timestamp, 'runBrowserManifest timestamp');
  const baseRuntimeConfig = cloneRuntimeConfig(getRuntimeConfig());
  const baseKernelPath = getActiveKernelPath();
  const baseKernelPathSource = getActiveKernelPathSource();
  const baseKernelPathPolicy = getActiveKernelPathPolicy();

  for (let i = 0; i < normalized.runs.length; i++) {
    const run = mergeRunDefaults(normalized.defaults, normalized.runs[i] || {});
    try {
      setRuntimeConfig(baseRuntimeConfig);
      setActiveKernelPath(baseKernelPath, baseKernelPathSource, baseKernelPathPolicy);
      await applyRuntimeForRun(run, options);
      const runTimestamp = resolveReportTimestamp(
        run.timestamp,
        `runBrowserManifest run[${i}] timestamp`,
        manifestTimestamp
      );
      const result = await runBrowserSuite({ ...run, timestamp: runTimestamp });
      results.push({
        ...result,
        label: run.label ?? `${run.workload || run.suite || 'inference'}:${result.modelId || 'unknown'}`,
      });
      options.onProgress?.({
        index: i + 1,
        total: normalized.runs.length,
        label: run.label ?? result.modelId ?? run.workload ?? run.suite ?? 'run',
      });
    } finally {
      setRuntimeConfig(baseRuntimeConfig);
      setActiveKernelPath(baseKernelPath, baseKernelPathSource, baseKernelPathPolicy);
    }
  }

  const summary = summarizeManifestRuns(results);
  const report = {
    timestamp: manifestTimestamp,
    summary,
    runs: results.map((result) => ({
      label: result.label,
      mode: result.mode,
      workload: result.workload,
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
