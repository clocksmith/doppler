
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
  modelSupportsEmbedding,
  modelSupportsRerank,
  modelSupportsSequence,
} from '../config/schema/manifest.schema.js';
import {
  resolveRuntime,
  loadRuntimeConfigFromUrl,
  applyRuntimeConfigFromUrl,
  loadRuntimeProfile,
  applyRuntimeProfile,
  applyRuntimeForRun,
} from './browser-harness/runtime-config.js';
import {
  cloneRuntimeConfig,
  runWithRuntimeIsolationForSuite,
} from './browser-harness/runtime-isolation.js';
import {
  resolveReportTimestamp,
  sanitizeReportOutput,
  normalizeManifest,
  mergeRunDefaults,
  summarizeManifestRuns,
} from './browser-harness/runtime-report.js';
import {
  buildSuiteSummary,
  normalizeCacheMode,
  normalizeLoadMode,
  normalizeWorkloadType,
} from './browser-harness/suite-summary.js';
import {
  assertDiffusionPerformanceArtifact,
} from './browser-harness/diffusion-performance.js';
import {
  toTimingNumber,
  buildFirstLoadComposition,
  safeToFixed,
  sampleTimingNumber,
  buildCanonicalTiming,
  buildLoadTimingDiagnostics,
  buildTimingDiagnostics,
} from './browser-harness/timing-diagnostics.js';
import { buildDecodeBottleneckDiagnostics } from './browser-harness/decode-diagnostics.js';
import {
  resolveDeviceInfo,
  resolveKernelPathForModel,
} from './browser-harness/model-resolution.js';
import { initializeSuiteModel } from './browser-harness/model-initialization.js';
import {
  resolveBenchmarkRunSettings,
} from './browser-harness/text-input.js';
import {
  normalizeDecodeRecordOpLabels,
  normalizeUniformCacheStats,
  buildDecodeRecordTopOps,
  buildDecodeRecordTopOpGroups,
  isCoherentOutput,
} from './browser-harness/text-evidence.js';
import {
  runEmbeddingSemanticChecks,
  runRerank,
  runRerankSemanticChecks,
  runSequenceEncoding,
  runTextInference,
  runEmbedding,
} from './browser-harness/text-execution.js';
import { buildSuiteContractMetrics } from './browser-harness-contract-helpers.js';
import {
  runDiffusionSuite,
  runEnergySuite,
} from './browser-harness-diffusion-energy-suites.js';
import { collectTrainingArtifactsFromSuiteResult } from './browser-harness-report-helpers.js';
import { sha256Hex } from '../formats/sha256.js';
import { stableSortObject } from '../formats/stable-sort-object.js';
import { assertCommandContextMatchesOptions } from '../tooling/command-context.js';
import {
  buildTokenCostLedger,
  isExecutionObservationRequested,
} from '../tooling/execution-cost-ledger.js';
import { createUnsupportedWorkloadError, hashStableJson, resolveDecodeCadence, resolveDispatchSuite, resolveExecutionGraphHash, resolveHarnessContext, resolveHarnessMode, resolveWorkload } from './browser-harness/request.js';
import { attachExecutionCostLedger, buildReferenceTranscriptSeed, resolvePipelineLoadTimings, serializeSequenceProbeRows, summarizePrefillRecordOps, summarizeTimingPhaseSamples } from './browser-harness/report.js';
import { runInferenceSuite, runKernelSuite, withHarnessPhase } from './browser-harness/inference-execution.js';
import { loadTrainingSuiteModule, runBenchSuite } from './browser-harness/benchmark-execution.js';
export { buildReferenceTranscriptSeed } from './browser-harness/report.js';
export { getBrowserSuiteDispatchMap, getBrowserSupportedSuites, resolveExecutionGraphHash } from './browser-harness/request.js';

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

async function dispatchBrowserSuite(mode, workload, options) {
  if (mode === 'verify' && workload === 'kernels') {
    return runKernelSuite(options);
  }
  if (mode === 'bench') {
    return runBenchSuite(options);
  }
  if (workload === 'embedding') {
    return runInferenceSuite({
      ...options,
      suiteName: 'embedding',
      expectedModelType: options.expectedModelType ?? 'embedding',
    });
  }
  if (workload === 'rerank') {
    return runInferenceSuite({
      ...options,
      suiteName: 'rerank',
      expectedModelType: options.expectedModelType ?? 'rerank',
    });
  }
  if (mode === 'verify' && workload === 'training') {
    return runTrainingSuite(options);
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

export async function runBrowserSuite(options = {}) {
  return runWithRuntimeIsolationForSuite(async () => {
    if (options.commandContext) {
      assertCommandContextMatchesOptions(options.commandContext, options);
    }
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
      modelId, surface: options.surface ?? 'browser',
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
    const requestReceipt = {
      ...(
        suiteResult.request && typeof suiteResult.request === 'object'
          ? suiteResult.request
          : {}
      ),
      runtimeProfile: options.runtimeProfile ?? null,
      runtimeConfigUrl: options.runtimeConfigUrl ?? null,
      runtimeConfig: cloneRuntimeConfig(getRuntimeConfig()),
    };
    return {
      ...suiteResult,
      mode,
      workload,
      request: requestReceipt,
      debugSnapshot,
      report,
      reportInfo,
    };
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
