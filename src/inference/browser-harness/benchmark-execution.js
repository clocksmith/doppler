import { getRuntimeConfig, setRuntimeConfig } from '../../config/runtime.js';
import { computeSampleStats } from '../../debug/stats.js';
import { validateTrainingMetricsReport } from '../../config/schema/training-metrics.schema.js';
import {
  modelSupportsEmbedding,
  modelSupportsRerank,
  modelSupportsSequence,
} from '../../config/schema/manifest.schema.js';
import {
  buildSuiteSummary,
  normalizeCacheMode,
  normalizeLoadMode,
  normalizeWorkloadType,
} from './suite-summary.js';
import {
  toTimingNumber,
  buildFirstLoadComposition,
  safeToFixed,
  sampleTimingNumber,
  buildCanonicalTiming,
  buildLoadTimingDiagnostics,
  buildTimingDiagnostics,
} from './timing-diagnostics.js';
import { buildDecodeBottleneckDiagnostics } from './decode-diagnostics.js';
import {
  resolveDeviceInfo,
  resolveKernelPathForModel,
} from './model-resolution.js';
import { initializeSuiteModel } from './model-initialization.js';
import {
  resolveBenchmarkRunSettings,
} from './text-input.js';
import {
  normalizeDecodeRecordOpLabels,
  normalizeUniformCacheStats,
  buildDecodeRecordTopOps,
  buildDecodeRecordTopOpGroups,
  isCoherentOutput,
} from './text-evidence.js';
import {
  runEmbeddingSemanticChecks,
  runRerank,
  runRerankSemanticChecks,
  runSequenceEncoding,
  runTextInference,
  runEmbedding,
} from './text-execution.js';
import { buildSuiteContractMetrics } from '../browser-harness-contract-helpers.js';
import {
  runDiffusionSuite,
  runEnergySuite,
} from '../browser-harness-diffusion-energy-suites.js';
import { createUnsupportedWorkloadError, hashStableJson, resolveDecodeCadence, resolveDispatchSuite, resolveExecutionGraphHash, resolveHarnessContext, resolveHarnessMode, resolveWorkload } from './request.js';
import { attachExecutionCostLedger, buildReferenceTranscriptSeed, resolvePipelineLoadTimings, serializeSequenceProbeRows, summarizePrefillRecordOps, summarizeTimingPhaseSamples } from './report.js';
import { runInferenceSuite, runKernelSuite, withHarnessPhase } from './inference-execution.js';

export const TRAINING_SUITE_MODULE_PATH = '../../experimental/training/suite.js';
import { runRerankBenchmark } from './benchmark-rerank.js';

export let trainingSuiteModulePromise = null;

export async function loadTrainingSuiteModule() {
  if (!trainingSuiteModulePromise) {
    trainingSuiteModulePromise = import(TRAINING_SUITE_MODULE_PATH);
  }
  return trainingSuiteModulePromise;
}

export async function runTrainingBenchSuite(options = {}) {
  const module = await loadTrainingSuiteModule();
  return module.runTrainingBenchSuite(options);
}

export function resolveBenchmarkIterationSettings(runtimeConfig) {
  const benchConfig = runtimeConfig?.shared?.benchmark?.run || {};
  return {
    warmupRuns: Math.max(0, Math.floor(benchConfig.warmupRuns ?? 0)),
    timedRuns: Math.max(1, Math.floor(benchConfig.timedRuns ?? 1)),
  };
}

export async function runBenchSuite(options = {}) {
  const startTime = performance.now();
  const runtimeConfig = getRuntimeConfig();
  const iterationSettings = resolveBenchmarkIterationSettings(runtimeConfig);
  const warmupRuns = iterationSettings.warmupRuns;
  const timedRuns = iterationSettings.timedRuns;
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
      benchRun: iterationSettings,
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

  if (workloadType === 'diffusion' || workloadType === 'diffusion_gemma') {
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
        workloadType,
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
  const benchRun = options.workload === 'rerank'
    ? iterationSettings
    : resolveBenchmarkRunSettings(
      runtimeConfig,
      harness.pipeline ?? harness,
      options.inferenceInput ?? null
    );
  const modelType = harness.manifest?.modelType || 'transformer';
  const supportsEmbedding = modelSupportsEmbedding(harness.manifest);
  const supportsRerank = modelSupportsRerank(harness.manifest);
  if (options.expectedModelType === 'embedding' && !supportsEmbedding) {
    throw new Error(
      `Expected an embedding-capable model for bench workload "${options.workload || 'inference'}", got modelType="${modelType}". ` +
      `Set inference.supportsEmbedding=true in the manifest for text-generation models that should expose pipeline.embed().`
    );
  }
  if (options.expectedModelType === 'rerank' && !supportsRerank) {
    throw new Error(
      `Expected a rerank-capable model for bench workload "${options.workload || 'inference'}", got modelType="${modelType}". ` +
      'Set inference.supportsRerank=true and inference.rerank in the manifest for models that should expose rerank scoring.'
    );
  }
  const safeModelLoadMs = toTimingNumber(harness.modelLoadMs, 0);

  let results;
  let metrics;
  let output = null;
  let timing;

  if (options.workload === 'rerank' && supportsRerank) {
    ({ metrics, output, results, timing } = await runRerankBenchmark({
      benchRun,
      cacheMode,
      harness,
      loadMode,
      runOptions: options,
      runtimeConfig,
      safeModelLoadMs,
      timedRuns,
      warmupRuns,
    }));
  } else if (modelType === 'embedding' || (options.workload === 'embedding' && supportsEmbedding)) {
    const durations = [];
    const timedDurations = [];
    const embeddingDims = [];
    const embeddingTokenCounts = [];
    const embeddingNorms = [];
    const embeddingPhases = [];
    let firstTimedEmbeddingMs = null;
    let invalidRuns = 0;
    let totalNonFiniteValues = 0;
    for (let i = 0; i < warmupRuns + timedRuns; i++) {
      harness.pipeline.reset?.();
      const run = await runEmbedding(harness.pipeline, runtimeConfig, {
        ...benchRun,
        benchmark: true,
      });
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
          if (run.phase && typeof run.phase === 'object') {
            embeddingPhases.push(run.phase);
          }
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
      phase: {
        ...summarizeTimingPhaseSamples(embeddingPhases, [
          'totalMs',
          'inputMs',
          'prefillMs',
          'submitWaitMs',
          'readbackMs',
          'decodeHiddenMs',
          'finalNormMs',
          'extractMs',
          'hiddenBytes',
          'tokens',
          'prefillRecordMs',
          'prefillRecordOps',
          'prefillRecordPasses',
          'prefillSubmitWaitMs',
          'gpuPrefillMs',
        ]),
        ...summarizePrefillRecordOps(embeddingPhases),
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
    const phaseTotals = [];
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
    const gpuDecodeRecordOps = [];
    const gpuDecodeRecordPasses = [];
    const gpuDecodeRecordMsPerOp = [];
    const gpuDecodeRecordMsPerPass = [];
    const gpuDecodeRecordPassesPerOp = [];
    const gpuDecodeRecordMsPerExecutedBatchToken = [];
    const gpuDecodeRecordOpsPerExecutedBatchToken = [];
    const gpuDecodeRecordPassesPerExecutedBatchToken = [];
    const gpuDecodeRecordOpLabels = {};
    let hasGpuDecodeRecordOpLabels = false;
    const gpuDecodeSubmitWaitMs = [];
    const gpuDecodeReadbackWaitMs = [];
    const gpuDecodeReadbackMapWaitMs = [];
    const gpuDecodeReadbackCleanupMs = [];
    const gpuDecodeReadbackCopyMs = [];
    const gpuDecodeOrchestrationMs = [];
    const gpuPrefillRecordMs = [];
    const gpuPrefillRecordOps = [];
    const gpuPrefillRecordPasses = [];
    const gpuPrefillRecordOpLabels = {};
    let hasGpuPrefillRecordOpLabels = false;
    const gpuPrefillSubmitWaitMs = [];
    const singleTokenSubmitWaitMs = [];
    const singleTokenReadbackWaitMs = [];
    const singleTokenReadbackMapWaitMs = [];
    const singleTokenReadbackCleanupMs = [];
    const singleTokenReadbackCopyMs = [];
    const singleTokenOrchestrationMs = [];
    const batchedForwardCalls = [];
    const unbatchedForwardCalls = [];
    const totalBatchedTimeMs = [];
    const totalUnbatchedTimeMs = [];
    const gpuSubmissions = [];
    const requestedBatchTokens = [];
    const effectiveBatchTokens = [];
    const executedBatchTokens = [];
    const resolvedBatchTokens = [];
    const maxBatchTokenCap = [];
    const batchClampCount = [];
    const plePreparedTokenCacheHits = [];
    const plePreparedTokenCacheMisses = [];
    const plePreparedTokenCacheEntries = [];
    const plePreparedTokenCacheBytes = [];

    let generatedText = null;
    let generatedPromptInput = null;
    let generatedReferenceTranscript = null;
    let lastPromptLabel = benchRun.promptLabel;
    let lastMaxTokens = benchRun.maxTokens;
    let lastDecodeMode = null;
    let lastBatchGuardReason = null;
    let lastExecutionPlan = null;
    let lastPrefillProfileSteps = null;
    let lastDecodeProfileSteps = null;
    let lastGpuUniformCache = null;
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
          benchmark: true,
          ...(options.inferenceInput ?? {}),
        })
      );
      if (i === warmupRuns + timedRuns - 1) {
        generatedText = run?.output ?? null;
        generatedPromptInput = run?.promptInput ?? null;
        generatedReferenceTranscript = buildReferenceTranscriptSeed(run, {
          executionGraphHash: resolveExecutionGraphHash(harness.manifest),
          kvCache: run?.phase?.kvCache ?? null,
        });
        lastPromptLabel = run?.prompt ?? benchRun.promptLabel;
        lastMaxTokens = Number.isFinite(run?.maxTokens) ? run.maxTokens : benchRun.maxTokens;
        lastDecodeMode = run?.phase?.decodeMode ?? null;
        lastBatchGuardReason = run?.phase?.batchGuardReason ?? null;
        lastExecutionPlan = run?.phase?.executionPlan ?? null;
        lastPrefillProfileSteps = run?.phase?.prefillProfileSteps ?? null;
        lastDecodeProfileSteps = run?.phase?.decodeProfileSteps ?? null;
      }
      if (i >= warmupRuns) {
        const phase = run?.phase ?? {};
        const phaseTokens = Array.isArray(run?.tokens) ? run.tokens : [];
        const phaseGpu = phase.gpu;
        const phaseBatching = phase.batching;
        const phasePlePreparedTokenCache = phase.plePreparedTokenCache;
        tokensPerSec.push(run?.tokensPerSec);
        durations.push(run?.durationMs);
        phaseTotals.push(phase.totalMs);
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
        const phaseGpuUniformCache = normalizeUniformCacheStats(phaseGpu?.uniformCache);
        if (phaseGpuUniformCache) {
          lastGpuUniformCache = phaseGpuUniformCache;
        }
        if (Number.isFinite(phaseGpu?.prefillMs)) gpuPrefillMs.push(phaseGpu.prefillMs);
        if (Number.isFinite(phaseGpu?.decodeMs)) gpuDecodeMs.push(phaseGpu.decodeMs);
        if (Number.isFinite(phaseGpu?.decodeRecordMs)) gpuDecodeRecordMs.push(phaseGpu.decodeRecordMs);
        if (Number.isFinite(phaseGpu?.decodeRecordOps)) gpuDecodeRecordOps.push(phaseGpu.decodeRecordOps);
        if (Number.isFinite(phaseGpu?.decodeRecordPasses)) gpuDecodeRecordPasses.push(phaseGpu.decodeRecordPasses);
        const phaseDecodeRecordOpLabels = normalizeDecodeRecordOpLabels(phaseGpu?.decodeRecordOpLabels);
        if (phaseDecodeRecordOpLabels) {
          hasGpuDecodeRecordOpLabels = true;
          for (const [label, count] of Object.entries(phaseDecodeRecordOpLabels)) {
            gpuDecodeRecordOpLabels[label] = (gpuDecodeRecordOpLabels[label] ?? 0) + count;
          }
        }
        if (Number.isFinite(phaseGpu?.decodeRecordMsPerOp)) {
          gpuDecodeRecordMsPerOp.push(phaseGpu.decodeRecordMsPerOp);
        }
        if (Number.isFinite(phaseGpu?.decodeRecordMsPerPass)) {
          gpuDecodeRecordMsPerPass.push(phaseGpu.decodeRecordMsPerPass);
        }
        if (Number.isFinite(phaseGpu?.decodeRecordPassesPerOp)) {
          gpuDecodeRecordPassesPerOp.push(phaseGpu.decodeRecordPassesPerOp);
        }
        if (Number.isFinite(phaseGpu?.decodeRecordMsPerExecutedBatchToken)) {
          gpuDecodeRecordMsPerExecutedBatchToken.push(phaseGpu.decodeRecordMsPerExecutedBatchToken);
        }
        if (Number.isFinite(phaseGpu?.decodeRecordOpsPerExecutedBatchToken)) {
          gpuDecodeRecordOpsPerExecutedBatchToken.push(phaseGpu.decodeRecordOpsPerExecutedBatchToken);
        }
        if (Number.isFinite(phaseGpu?.decodeRecordPassesPerExecutedBatchToken)) {
          gpuDecodeRecordPassesPerExecutedBatchToken.push(phaseGpu.decodeRecordPassesPerExecutedBatchToken);
        }
        if (Number.isFinite(phaseGpu?.decodeSubmitWaitMs)) gpuDecodeSubmitWaitMs.push(phaseGpu.decodeSubmitWaitMs);
        if (Number.isFinite(phaseGpu?.decodeReadbackWaitMs)) gpuDecodeReadbackWaitMs.push(phaseGpu.decodeReadbackWaitMs);
        if (Number.isFinite(phaseGpu?.decodeReadbackMapWaitMs)) gpuDecodeReadbackMapWaitMs.push(phaseGpu.decodeReadbackMapWaitMs);
        if (Number.isFinite(phaseGpu?.decodeReadbackCleanupMs)) gpuDecodeReadbackCleanupMs.push(phaseGpu.decodeReadbackCleanupMs);
        if (Number.isFinite(phaseGpu?.decodeReadbackCopyMs)) gpuDecodeReadbackCopyMs.push(phaseGpu.decodeReadbackCopyMs);
        if (Number.isFinite(phaseGpu?.prefillRecordMs)) gpuPrefillRecordMs.push(phaseGpu.prefillRecordMs);
        if (Number.isFinite(phaseGpu?.prefillRecordOps)) gpuPrefillRecordOps.push(phaseGpu.prefillRecordOps);
        if (Number.isFinite(phaseGpu?.prefillRecordPasses)) gpuPrefillRecordPasses.push(phaseGpu.prefillRecordPasses);
        const phasePrefillRecordOpLabels = normalizeDecodeRecordOpLabels(phaseGpu?.prefillRecordOpLabels);
        if (phasePrefillRecordOpLabels) {
          hasGpuPrefillRecordOpLabels = true;
          for (const [label, count] of Object.entries(phasePrefillRecordOpLabels)) {
            gpuPrefillRecordOpLabels[label] = (gpuPrefillRecordOpLabels[label] ?? 0) + count;
          }
        }
        if (Number.isFinite(phaseGpu?.prefillSubmitWaitMs)) gpuPrefillSubmitWaitMs.push(phaseGpu.prefillSubmitWaitMs);
        if (Number.isFinite(phaseGpu?.decodeOrchestrationMs)) {
          gpuDecodeOrchestrationMs.push(phaseGpu.decodeOrchestrationMs);
        }
        if (Number.isFinite(phaseGpu?.singleTokenSubmitWaitMs)) singleTokenSubmitWaitMs.push(phaseGpu.singleTokenSubmitWaitMs);
        if (Number.isFinite(phaseGpu?.singleTokenReadbackWaitMs)) singleTokenReadbackWaitMs.push(phaseGpu.singleTokenReadbackWaitMs);
        if (Number.isFinite(phaseGpu?.singleTokenReadbackMapWaitMs)) singleTokenReadbackMapWaitMs.push(phaseGpu.singleTokenReadbackMapWaitMs);
        if (Number.isFinite(phaseGpu?.singleTokenReadbackCleanupMs)) singleTokenReadbackCleanupMs.push(phaseGpu.singleTokenReadbackCleanupMs);
        if (Number.isFinite(phaseGpu?.singleTokenReadbackCopyMs)) singleTokenReadbackCopyMs.push(phaseGpu.singleTokenReadbackCopyMs);
        if (Number.isFinite(phaseGpu?.singleTokenOrchestrationMs)) singleTokenOrchestrationMs.push(phaseGpu.singleTokenOrchestrationMs);
        if (Number.isFinite(phaseBatching?.batchedForwardCalls)) batchedForwardCalls.push(phaseBatching.batchedForwardCalls);
        if (Number.isFinite(phaseBatching?.unbatchedForwardCalls)) unbatchedForwardCalls.push(phaseBatching.unbatchedForwardCalls);
        if (Number.isFinite(phaseBatching?.totalBatchedTimeMs)) totalBatchedTimeMs.push(phaseBatching.totalBatchedTimeMs);
        if (Number.isFinite(phaseBatching?.totalUnbatchedTimeMs)) totalUnbatchedTimeMs.push(phaseBatching.totalUnbatchedTimeMs);
        if (Number.isFinite(phaseBatching?.gpuSubmissions)) gpuSubmissions.push(phaseBatching.gpuSubmissions);
        if (Number.isFinite(phaseBatching?.requestedBatchTokens)) requestedBatchTokens.push(phaseBatching.requestedBatchTokens);
        if (Number.isFinite(phaseBatching?.effectiveBatchTokens)) effectiveBatchTokens.push(phaseBatching.effectiveBatchTokens);
        if (Number.isFinite(phaseBatching?.executedBatchTokens)) executedBatchTokens.push(phaseBatching.executedBatchTokens);
        if (Number.isFinite(phaseBatching?.resolvedBatchTokens)) resolvedBatchTokens.push(phaseBatching.resolvedBatchTokens);
        if (Number.isFinite(phaseBatching?.maxBatchTokenCap)) maxBatchTokenCap.push(phaseBatching.maxBatchTokenCap);
        if (Number.isFinite(phaseBatching?.batchClampCount)) batchClampCount.push(phaseBatching.batchClampCount);
        if (Number.isFinite(phasePlePreparedTokenCache?.hits)) plePreparedTokenCacheHits.push(phasePlePreparedTokenCache.hits);
        if (Number.isFinite(phasePlePreparedTokenCache?.misses)) plePreparedTokenCacheMisses.push(phasePlePreparedTokenCache.misses);
        if (Number.isFinite(phasePlePreparedTokenCache?.entries)) plePreparedTokenCacheEntries.push(phasePlePreparedTokenCache.entries);
        if (Number.isFinite(phasePlePreparedTokenCache?.bytes)) plePreparedTokenCacheBytes.push(phasePlePreparedTokenCache.bytes);
      }
    }

    const totalMsStats = computeSampleStats(phaseTotals);
    const wallRunMsStats = computeSampleStats(durations);
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
    const gpuDecodeRecordOpsStats = computeSampleStats(gpuDecodeRecordOps);
    const gpuDecodeRecordPassesStats = computeSampleStats(gpuDecodeRecordPasses);
    const gpuPrefillRecordOpsStats = computeSampleStats(gpuPrefillRecordOps);
    const gpuPrefillRecordPassesStats = computeSampleStats(gpuPrefillRecordPasses);
    const gpuDecodeRecordOpLabelSampleCount = gpuDecodeRecordOps.length > 0
      ? gpuDecodeRecordOps.length
      : 1;
    const gpuDecodeRecordMeanOpLabels = {};
    if (hasGpuDecodeRecordOpLabels) {
      for (const [label, count] of Object.entries(gpuDecodeRecordOpLabels)) {
        gpuDecodeRecordMeanOpLabels[label] = count / gpuDecodeRecordOpLabelSampleCount;
      }
    }
    const gpuPrefillRecordOpLabelSampleCount = gpuPrefillRecordOps.length > 0
      ? gpuPrefillRecordOps.length
      : 1;
    const gpuPrefillRecordMeanOpLabels = {};
    if (hasGpuPrefillRecordOpLabels) {
      for (const [label, count] of Object.entries(gpuPrefillRecordOpLabels)) {
        gpuPrefillRecordMeanOpLabels[label] = count / gpuPrefillRecordOpLabelSampleCount;
      }
    }
    const hasGpuStats = gpuPrefillMs.length > 0 || gpuDecodeMs.length > 0 || gpuDecodeRecordMs.length > 0
      || gpuDecodeRecordOps.length > 0 || gpuDecodeRecordPasses.length > 0
      || gpuPrefillRecordOps.length > 0 || gpuPrefillRecordPasses.length > 0
      || gpuDecodeRecordMsPerOp.length > 0 || gpuDecodeRecordMsPerPass.length > 0
      || gpuDecodeRecordPassesPerOp.length > 0
      || gpuDecodeRecordMsPerExecutedBatchToken.length > 0
      || gpuDecodeRecordOpsPerExecutedBatchToken.length > 0
      || gpuDecodeRecordPassesPerExecutedBatchToken.length > 0
      || hasGpuDecodeRecordOpLabels
      || hasGpuPrefillRecordOpLabels
      || gpuDecodeSubmitWaitMs.length > 0 || gpuDecodeReadbackWaitMs.length > 0
      || gpuDecodeReadbackMapWaitMs.length > 0 || gpuDecodeReadbackCleanupMs.length > 0
      || gpuDecodeReadbackCopyMs.length > 0
      || gpuDecodeOrchestrationMs.length > 0
      || lastGpuUniformCache
      || singleTokenSubmitWaitMs.length > 0 || singleTokenReadbackWaitMs.length > 0
      || singleTokenReadbackMapWaitMs.length > 0 || singleTokenReadbackCleanupMs.length > 0
      || singleTokenReadbackCopyMs.length > 0
      || singleTokenOrchestrationMs.length > 0;
    const gpuPhaseStats = hasGpuStats
      ? {
        prefillMs: computeSampleStats(gpuPrefillMs),
        decodeMs: computeSampleStats(gpuDecodeMs),
        decodeRecordMs: computeSampleStats(gpuDecodeRecordMs),
        decodeRecordOps: gpuDecodeRecordOpsStats,
        decodeRecordPasses: gpuDecodeRecordPassesStats,
        decodeRecordMsPerOp: computeSampleStats(gpuDecodeRecordMsPerOp),
        decodeRecordMsPerPass: computeSampleStats(gpuDecodeRecordMsPerPass),
        decodeRecordPassesPerOp: computeSampleStats(gpuDecodeRecordPassesPerOp),
        decodeRecordMsPerExecutedBatchToken: computeSampleStats(gpuDecodeRecordMsPerExecutedBatchToken),
        decodeRecordOpsPerExecutedBatchToken: computeSampleStats(gpuDecodeRecordOpsPerExecutedBatchToken),
        decodeRecordPassesPerExecutedBatchToken: computeSampleStats(gpuDecodeRecordPassesPerExecutedBatchToken),
        decodeRecordUniqueOpLabels: hasGpuDecodeRecordOpLabels ? Object.keys(gpuDecodeRecordOpLabels).length : null,
        decodeRecordTopOps: hasGpuDecodeRecordOpLabels
          ? buildDecodeRecordTopOps(
            gpuDecodeRecordMeanOpLabels,
            gpuDecodeRecordOpsStats?.mean
          )
          : [],
        decodeRecordTopOpGroups: hasGpuDecodeRecordOpLabels
          ? buildDecodeRecordTopOpGroups(
            gpuDecodeRecordMeanOpLabels,
            gpuDecodeRecordOpsStats?.mean
          )
          : [],
        decodeSubmitWaitMs: computeSampleStats(gpuDecodeSubmitWaitMs),
        decodeReadbackWaitMs: computeSampleStats(gpuDecodeReadbackWaitMs),
        decodeReadbackMapWaitMs: computeSampleStats(gpuDecodeReadbackMapWaitMs),
        decodeReadbackCleanupMs: computeSampleStats(gpuDecodeReadbackCleanupMs),
        decodeReadbackCopyMs: computeSampleStats(gpuDecodeReadbackCopyMs),
        decodeOrchestrationMs: computeSampleStats(gpuDecodeOrchestrationMs),
        prefillRecordMs: computeSampleStats(gpuPrefillRecordMs),
        prefillRecordOps: gpuPrefillRecordOpsStats,
        prefillRecordPasses: gpuPrefillRecordPassesStats,
        prefillRecordUniqueOpLabels: hasGpuPrefillRecordOpLabels ? Object.keys(gpuPrefillRecordOpLabels).length : null,
        prefillRecordTopOps: hasGpuPrefillRecordOpLabels
          ? buildDecodeRecordTopOps(
            gpuPrefillRecordMeanOpLabels,
            gpuPrefillRecordOpsStats?.mean
          )
          : [],
        prefillRecordTopOpGroups: hasGpuPrefillRecordOpLabels
          ? buildDecodeRecordTopOpGroups(
            gpuPrefillRecordMeanOpLabels,
            gpuPrefillRecordOpsStats?.mean
          )
          : [],
        prefillSubmitWaitMs: computeSampleStats(gpuPrefillSubmitWaitMs),
        uniformCache: lastGpuUniformCache,
        singleTokenSubmitWaitMs: computeSampleStats(singleTokenSubmitWaitMs),
        singleTokenReadbackWaitMs: computeSampleStats(singleTokenReadbackWaitMs),
        singleTokenReadbackMapWaitMs: computeSampleStats(singleTokenReadbackMapWaitMs),
        singleTokenReadbackCleanupMs: computeSampleStats(singleTokenReadbackCleanupMs),
        singleTokenReadbackCopyMs: computeSampleStats(singleTokenReadbackCopyMs),
        singleTokenOrchestrationMs: computeSampleStats(singleTokenOrchestrationMs),
      }
      : null;
    const hasBatchingStats = batchedForwardCalls.length > 0
      || unbatchedForwardCalls.length > 0
      || totalBatchedTimeMs.length > 0
      || totalUnbatchedTimeMs.length > 0
      || gpuSubmissions.length > 0
      || requestedBatchTokens.length > 0
      || effectiveBatchTokens.length > 0
      || executedBatchTokens.length > 0
      || resolvedBatchTokens.length > 0
      || maxBatchTokenCap.length > 0
      || batchClampCount.length > 0;
    const batchingPhaseStats = hasBatchingStats
      ? {
        batchedForwardCalls: computeSampleStats(batchedForwardCalls),
        unbatchedForwardCalls: computeSampleStats(unbatchedForwardCalls),
        totalBatchedTimeMs: computeSampleStats(totalBatchedTimeMs),
        totalUnbatchedTimeMs: computeSampleStats(totalUnbatchedTimeMs),
        gpuSubmissions: computeSampleStats(gpuSubmissions),
        requestedBatchTokens: computeSampleStats(requestedBatchTokens),
        effectiveBatchTokens: computeSampleStats(effectiveBatchTokens),
        executedBatchTokens: computeSampleStats(executedBatchTokens),
        resolvedBatchTokens: computeSampleStats(resolvedBatchTokens),
        maxBatchTokenCap: computeSampleStats(maxBatchTokenCap),
        batchClampCount: computeSampleStats(batchClampCount),
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
      prompt: lastPromptLabel,
      maxTokens: lastMaxTokens,
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
      wallRunMs: sampleTimingNumber(wallRunMsStats, 'median'),
      modelLoadMs: safeModelLoadMs,
      throughput: {
        tokensPerSec: tokensPerSecStats,
        prefillTokensPerSec: prefillTokensPerSecStats,
        prefillTokensPerSecTtft: prefillTokensPerSecTtftStats,
        decodeTokensPerSec: decodeTokensPerSecStats,
      },
      latency: {
        totalMs: totalMsStats,
        wallRunMs: wallRunMsStats,
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
      decodeCadence: resolveDecodeCadence(getRuntimeConfig(), lastExecutionPlan),
      plePreparedTokenCache: plePreparedTokenCacheStats,
      decodeMode: lastDecodeMode,
      batchGuardReason: lastBatchGuardReason,
      executionPlan: lastExecutionPlan,
      prefillProfileSteps: lastPrefillProfileSteps,
      decodeProfileSteps: lastDecodeProfileSteps,
      generatedText,
      referenceTranscript: generatedReferenceTranscript,
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
  const loadTimings = resolvePipelineLoadTimings(harness.pipeline);
  const loadDiagnostics = buildLoadTimingDiagnostics(
    safeModelLoadMs,
    loadTimings.loadTiming,
    loadTimings.pipelineLoadTiming
  );
  const decodeBottleneck = buildDecodeBottleneckDiagnostics(metrics, timing);
  if (decodeBottleneck) {
    metrics.decodeBottleneck = decodeBottleneck;
  }

  if (typeof harness.pipeline.unload === 'function' && !options.keepPipeline) {
    await harness.pipeline.unload();
  }

  const summary = buildSuiteSummary('bench', results, startTime);
  const timingDiagnostics = buildTimingDiagnostics(timing, {
    source: 'doppler',
    prefillSemantics: 'internal_prefill_phase',
    loadTiming: loadTimings.loadTiming,
    pipelineLoadTiming: loadTimings.pipelineLoadTiming,
  });
  if (decodeBottleneck) {
    timingDiagnostics.decodeBottleneck = decodeBottleneck;
  }
  const firstLoad = buildFirstLoadComposition({
    modelLoadMs: timing.modelLoadMs,
    firstTokenMs: timing.firstTokenMs,
    firstResponseMs: timing.firstResponseMs,
  });
  const metricsWithContracts = attachExecutionCostLedger(buildSuiteContractMetrics(
    'bench',
    loadDiagnostics ? { ...metrics, load: loadDiagnostics } : metrics,
    harness.manifest
  ), runtimeConfig, harness.manifest, { force: true });
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
