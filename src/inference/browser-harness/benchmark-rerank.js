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

export async function runRerankBenchmark(options) {
  const {
    benchRun,
    cacheMode,
    harness,
    loadMode,
    runOptions,
    runtimeConfig,
    safeModelLoadMs,
    timedRuns,
    warmupRuns,
  } = options;

    const durations = [];
    const timedDurations = [];
    const documentCounts = [];
    const topDocumentScores = [];
    const topDocumentProbabilities = [];
    const rerankPhases = [];
    const rerankDocumentPhases = [];
    let invalidRuns = 0;
    let nonFiniteScores = 0;
    let lastRun = null;

    for (let i = 0; i < warmupRuns + timedRuns; i++) {
      harness.pipeline.reset?.();
      const run = await runRerank(harness.pipeline, runtimeConfig, {
        ...benchRun,
        benchmark: true,
      });
      if (i >= warmupRuns) {
        timedDurations.push(run.durationMs);
        const finiteScores = run.scores.filter((entry) => (
          Number.isFinite(entry.score)
          && Number.isFinite(entry.probability)
          && Number.isFinite(entry.trueLogit)
          && Number.isFinite(entry.falseLogit)
        ));
        nonFiniteScores += run.scores.length - finiteScores.length;
        const hasRanking = Array.isArray(run.ranking) && run.ranking.length === run.documentCount;
        if (finiteScores.length === run.scores.length && hasRanking && run.documentCount > 0) {
          durations.push(run.durationMs);
          if (run.phase && typeof run.phase === 'object') {
            rerankPhases.push(run.phase);
            if (Array.isArray(run.phase.documents)) {
              for (const documentPhase of run.phase.documents) {
                if (documentPhase?.phase && typeof documentPhase.phase === 'object') {
                  rerankDocumentPhases.push(documentPhase.phase);
                }
              }
            }
          }
          documentCounts.push(run.documentCount);
          if (Number.isFinite(run.topDocument?.score)) {
            topDocumentScores.push(run.topDocument.score);
          }
          if (Number.isFinite(run.topDocument?.probability)) {
            topDocumentProbabilities.push(run.topDocument.probability);
          }
        } else {
          invalidRuns++;
        }
        lastRun = run;
      }
    }

    const semantic = await runRerankSemanticChecks(harness.pipeline, runOptions);
    const rerankMsStats = computeSampleStats(durations);
    const timedRerankMsStats = computeSampleStats(timedDurations);
    const documentCountStats = computeSampleStats(documentCounts);
    const topScoreStats = computeSampleStats(topDocumentScores);
    const topProbabilityStats = computeSampleStats(topDocumentProbabilities);
    const avgMs = rerankMsStats.mean;
    const semanticPassed = semantic.passed;
    const rerankPrefixPhases = rerankPhases
      .map((phase) => phase?.prefix)
      .filter((phase) => phase && typeof phase === 'object');

    const results = [
      {
        name: 'benchmark-rerank',
        passed: durations.length > 0 && invalidRuns === 0,
        duration: durations.reduce((sum, value) => sum + value, 0),
        error: durations.length > 0
          ? (
            invalidRuns === 0
              ? undefined
              : `Invalid rerank runs: ${invalidRuns} (non-finite scores or incomplete ranking observed)`
          )
          : 'No valid rerank benchmark runs completed',
      },
      {
        name: 'benchmark-rerank-semantic',
        passed: semanticPassed,
        duration: semantic.durationMs,
        error: semanticPassed
          ? undefined
          : (
            `Rerank semantic checks below threshold: pairs=${(semantic.pairAcc * 100).toFixed(1)}% `
            + `(min ${(semantic.minPairAcc * 100).toFixed(1)}%). `
            + (semantic.failedCaseIds.length > 0 ? `Failed: ${semantic.failedCaseIds.join(', ')}` : '')
          ),
      },
    ];

    const output = {
      mode: 'rerank',
      query: lastRun?.query ?? null,
      documentCount: lastRun?.documentCount ?? null,
      topDocument: lastRun?.topDocument ?? null,
      ranking: lastRun?.ranking ?? [],
      semantic: {
        passed: semanticPassed,
        pairAcc: Number(semantic.pairAcc.toFixed(4)),
        failedCaseIds: semantic.failedCaseIds,
        details: {
          pairs: semantic.pairs,
        },
      },
    };

    const metrics = {
      warmupRuns,
      timedRuns,
      validRuns: durations.length,
      invalidRuns,
      invalidRatePct: Number((timedRuns > 0 ? (invalidRuns / timedRuns) * 100 : 0).toFixed(2)),
      query: lastRun?.query ?? null,
      documentCount: Math.round(documentCountStats.mean),
      topDocumentIndex: lastRun?.topDocument?.index ?? null,
      topDocumentScore: lastRun?.topDocument?.score == null ? null : Number(lastRun.topDocument.score.toFixed(6)),
      topDocumentProbability: lastRun?.topDocument?.probability == null ? null : Number(lastRun.topDocument.probability.toFixed(6)),
      topDocumentScoreStats: topScoreStats,
      topDocumentProbabilityStats: topProbabilityStats,
      nonFiniteScores,
      firstTimedRerankMs: Number((timedDurations[0] ?? 0).toFixed(2)),
      minRerankMs: Number(rerankMsStats.min.toFixed(2)),
      medianRerankMs: Number(rerankMsStats.median.toFixed(2)),
      p95RerankMs: Number(rerankMsStats.p95.toFixed(2)),
      p99RerankMs: Number(rerankMsStats.p99.toFixed(2)),
      maxRerankMs: Number(rerankMsStats.max.toFixed(2)),
      stdDevRerankMs: Number(rerankMsStats.stdDev.toFixed(2)),
      ci95RerankMs: Number(rerankMsStats.ci95.toFixed(2)),
      avgRerankMs: Number(avgMs.toFixed(2)),
      avgReranksPerSec: Number((avgMs > 0 ? (1000 / avgMs) : 0).toFixed(2)),
      semanticPassed,
      semanticDurationMs: Number(semantic.durationMs.toFixed(2)),
      semanticPairAcc: Number(semantic.pairAcc.toFixed(4)),
      semanticPairPassed: semantic.pairPassed,
      semanticPairTotal: semantic.pairTotal,
      semanticMinPairAcc: Number(semantic.minPairAcc.toFixed(4)),
      semanticMinScoreMargin: Number(semantic.minScoreMargin.toFixed(4)),
      semanticFailedCases: semantic.failedCaseIds,
      semanticDetails: {
        pairs: semantic.pairs,
      },
      modelLoadMs: safeModelLoadMs,
      latency: {
        timedRerankMs: timedRerankMsStats,
        rerankMs: rerankMsStats,
      },
      phase: summarizeTimingPhaseSamples(rerankPhases, [
        'totalMs',
        'prefixMs',
        'prefixTokens',
        'documentCount',
        'documentTotalMs',
        'maxDocumentMs',
        'avgDocumentMs',
        'prefix.totalMs',
        'prefix.prefillMs',
        'prefix.prefillRecordMs',
        'prefix.prefillRecordOps',
        'prefix.prefillRecordPasses',
        'prefix.prefillSubmitWaitMs',
      ]),
      prefixPhase: {
        ...summarizeTimingPhaseSamples(rerankPrefixPhases, [
          'totalMs',
          'prefillMs',
          'prefillRecordMs',
          'prefillRecordOps',
          'prefillRecordPasses',
          'prefillSubmitWaitMs',
          'gpuPrefillMs',
          'tokens',
        ]),
        ...summarizePrefillRecordOps(rerankPrefixPhases),
      },
      documentPhase: {
        ...summarizeTimingPhaseSamples(rerankDocumentPhases, [
          'totalMs',
          'prefillCallMs',
          'inputMs',
          'prefillMs',
          'prefillRecordMs',
          'prefillRecordOps',
          'prefillRecordPasses',
          'prefillSubmitWaitMs',
          'gpuPrefillMs',
          'tokens',
          'selectedTokenCount',
          'prefixTokens',
          'suffixTokens',
          'promptChars',
        ]),
        ...summarizePrefillRecordOps(rerankDocumentPhases),
      },
      documents: {
        count: documentCountStats,
      },
    };

    const timing = buildCanonicalTiming({
      modelLoadMs: safeModelLoadMs,
      firstTokenMs: null,
      firstResponseMs: Number.isFinite(timedDurations[0])
        ? safeModelLoadMs + timedDurations[0]
        : null,
      prefillMs: null,
      decodeMs: null,
      totalRunMs: rerankMsStats.median,
      cacheMode,
      loadMode,
    });

  return { metrics, output, results, timing };
}
