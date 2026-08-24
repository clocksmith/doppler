import { getRuntimeConfig, setRuntimeConfig } from '../../config/runtime.js';
import {
  setActiveKernelPath,
  getActiveKernelPath,
  getActiveKernelPathSource,
  getActiveKernelPathPolicy,
} from '../../config/kernel-path-loader.js';
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
import { createUnsupportedWorkloadError, hashStableJson, resolveDecodeCadence, resolveDispatchSuite, resolveExecutionGraphHash, resolveHarnessContext, resolveHarnessMode, resolveWorkload } from './request.js';
import { attachExecutionCostLedger, buildReferenceTranscriptSeed, resolvePipelineLoadTimings, serializeSequenceProbeRows, summarizePrefillRecordOps, summarizeTimingPhaseSamples } from './report.js';

export function allFinite(values) {
  if (!ArrayBuffer.isView(values)) {
    return false;
  }
  for (const value of values) {
    if (!Number.isFinite(value)) {
      return false;
    }
  }
  return true;
}

export async function runKernelSuite(options = {}) {
  const startTime = performance.now();
  const { testHarness, initGPU } = await import('../../../tests/kernels/browser/test-page.js');
  const { runKernelSuite: runAllKernelTests } = await import('../../../tests/kernels/browser/kernel-suite.js');
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

export async function runInferenceSuite(options = {}) {
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
  const supportsEmbedding = modelSupportsEmbedding(harness.manifest);
  const supportsRerank = modelSupportsRerank(harness.manifest);
  if (options.expectedModelType === 'embedding' && !supportsEmbedding) {
    throw new Error(
      `Expected an embedding-capable model for workload "${options.workload || 'inference'}", got modelType="${modelType}". ` +
      `Set inference.supportsEmbedding=true in the manifest for text-generation models that should expose pipeline.embed().`
    );
  }
  if (options.expectedModelType === 'rerank' && !supportsRerank) {
    throw new Error(
      `Expected a rerank-capable model for workload "${options.workload || 'inference'}", got modelType="${modelType}". ` +
      'Set inference.supportsRerank=true and inference.rerank in the manifest for models that should expose rerank scoring.'
    );
  }
  const safeModelLoadMs = toTimingNumber(harness.modelLoadMs, 0);

  let results;
  let output = null;
  let metrics;

  if (options.workload === 'rerank' && supportsRerank) {
    const run = await runRerank(harness.pipeline, runtimeConfig);
    const semantic = await runRerankSemanticChecks(harness.pipeline, options);
    const allScoresFinite = run.scores.every((entry) => (
      Number.isFinite(entry.score)
      && Number.isFinite(entry.probability)
      && Number.isFinite(entry.trueLogit)
      && Number.isFinite(entry.falseLogit)
    ));
    const hasRanking = Array.isArray(run.ranking) && run.ranking.length === run.documentCount;
    const isValidRerank = allScoresFinite && hasRanking && run.documentCount > 0;
    const isSemanticValid = semantic.passed;
    output = {
      mode: 'rerank',
      query: run.query,
      documentCount: run.documentCount,
      topDocument: run.topDocument,
      ranking: run.ranking,
      semantic: {
        passed: isSemanticValid,
        pairAcc: Number(semantic.pairAcc.toFixed(4)),
        failedCaseIds: semantic.failedCaseIds,
        details: {
          pairs: semantic.pairs,
        },
      },
    };
    results = [
      {
        name: 'rerank',
        passed: isValidRerank,
        duration: run.durationMs,
        error: isValidRerank
          ? undefined
          : 'Rerank scores must be finite and produce a full ranking.',
      },
      {
        name: 'rerank-semantic',
        passed: isSemanticValid,
        duration: semantic.durationMs,
        error: isSemanticValid
          ? undefined
          : (
            `Rerank semantic checks below threshold: pairs=${(semantic.pairAcc * 100).toFixed(1)}% `
            + `(min ${(semantic.minPairAcc * 100).toFixed(1)}%). `
            + (semantic.failedCaseIds.length > 0 ? `Failed: ${semantic.failedCaseIds.join(', ')}` : '')
          ),
      },
    ];
    metrics = {
      query: run.query,
      documentCount: run.documentCount,
      topDocumentIndex: run.topDocument?.index ?? null,
      topDocumentScore: run.topDocument?.score == null ? null : Number(run.topDocument.score.toFixed(6)),
      topDocumentProbability: run.topDocument?.probability == null ? null : Number(run.topDocument.probability.toFixed(6)),
      rerankMs: Number(run.durationMs.toFixed(2)),
      rerankRanking: run.ranking,
      semanticPassed: isSemanticValid,
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
      endToEndMs: safeToFixed(safeModelLoadMs + run.durationMs),
    };
  } else if (options.inferenceInput?.sequence != null) {
    if (!modelSupportsSequence(harness.manifest)) {
      throw new Error(
        `Model "${harness.manifest?.modelId || options.modelId || 'unknown'}" does not declare sequence encoding support.`
      );
    }
    const declaredAlphabet = harness.manifest?.inference?.sequence?.alphabet ?? null;
    if (
      options.inferenceInput.sequenceAlphabet != null
      && options.inferenceInput.sequenceAlphabet !== declaredAlphabet
    ) {
      throw new Error(
        `Sequence alphabet mismatch: request declares "${options.inferenceInput.sequenceAlphabet}" but model declares "${declaredAlphabet}".`
      );
    }
    const run = await runSequenceEncoding(harness.pipeline, options.inferenceInput);
    const pooledEmbeddingFinite = allFinite(run.pooledEmbedding);
    const tokenEmbeddingsFinite = run.tokenEmbeddings == null || allFinite(run.tokenEmbeddings);
    const logitsFinite = run.logits == null || allFinite(run.logits);
    const isValidSequence = Array.isArray(run.tokens)
      && run.tokens.length > 0
      && Number.isInteger(run.embeddingDim)
      && run.embeddingDim > 0
      && pooledEmbeddingFinite
      && tokenEmbeddingsFinite
      && logitsFinite;
    output = {
      mode: 'sequence',
      model: {
        modelId: harness.manifest?.modelId ?? null,
        sourceCheckpointId: harness.manifest?.artifactIdentity?.sourceCheckpointId ?? null,
      },
      input: {
        sequence: run.sequence,
        alphabet: run.alphabet,
      },
      tokens: Array.from(run.tokens ?? []),
      tokenMask: Array.from(run.tokenMask ?? []),
      includedTokenCount: run.includedTokenCount,
      embeddingDim: run.embeddingDim,
      vocabSize: run.vocabSize,
      pooledEmbedding: Array.from(run.pooledEmbedding ?? []),
      tokenEmbeddingProbes: serializeSequenceProbeRows(
        run.tokenEmbeddings,
        run.embeddingDim,
        options.inferenceInput.probePositions
      ),
      logits: run.logits == null ? null : Array.from(run.logits),
      finite: {
        pooledEmbedding: pooledEmbeddingFinite,
        tokenEmbeddings: tokenEmbeddingsFinite,
        logits: logitsFinite,
      },
    };
    results = [
      {
        name: 'sequence-encoding',
        passed: isValidSequence,
        duration: run.durationMs,
        error: isValidSequence
          ? undefined
          : 'Sequence encoding must return finite pooled, requested token, and requested logit outputs.',
      },
    ];
    metrics = {
      sequence: run.sequence,
      sequenceAlphabet: run.alphabet,
      sequenceTokens: run.tokens?.length ?? 0,
      sequenceEmbeddingDim: run.embeddingDim,
      sequenceIncludedTokenCount: run.includedTokenCount,
      sequenceTokenEmbeddingsRequested: run.tokenEmbeddings != null,
      sequenceLogitsRequested: run.logits != null,
      sequenceEncodingMs: Number(run.durationMs.toFixed(2)),
      sequenceFinite: output.finite,
      modelLoadMs: safeModelLoadMs,
      endToEndMs: safeToFixed(safeModelLoadMs + run.durationMs),
    };
  } else if (modelType === 'embedding' || (options.workload === 'embedding' && supportsEmbedding)) {
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
      wallRunMs: safeToFixed(run.phase.wallMs ?? run.durationMs),
      prefillTokens: Math.round(run.phase.prefillTokens),
      decodeTokens: Math.round(run.phase.decodeTokens),
      stopReason: run.phase.stopReason ?? null,
      stopTokenId: Number.isInteger(run.phase.stopTokenId) ? run.phase.stopTokenId : null,
      prefillTokensPerSec: safeToFixed(run.phase.prefillTokensPerSec),
      prefillTokensPerSecTtft: safeToFixed(run.phase.prefillTokensPerSecTtft),
      decodeTokensPerSec: safeToFixed(run.phase.decodeTokensPerSec),
      modelLoadMs: safeModelLoadMs,
      gpu: run.phase.gpu,
      batching: run.phase.batching ?? null,
      plePreparedTokenCache: run.phase.plePreparedTokenCache ?? null,
      prefillProfileSteps: run.phase.prefillProfileSteps,
      decodeProfileSteps: run.phase.decodeProfileSteps,
      executionPlan: run.phase.executionPlan,
      kernelPathId: run.phase.kernelPathId,
      kernelPathSource: run.phase.kernelPathSource,
      generationDiagnostics: run.tokenDiagnostics, generationConfig: run.generationConfig,
      initialExecutionIdentity: run.initialExecutionIdentity,
      kvCache: run.phase.kvCache ?? null,
      referenceTranscript: buildReferenceTranscriptSeed(run, {
        executionGraphHash: resolveExecutionGraphHash(harness.manifest),
        kvCache: run.phase.kvCache ?? null, surface: options.surface === 'node' ? 'node-webgpu' : 'browser-webgpu',
      }),
      operatorDiagnostics: run.phase.operatorDiagnostics ?? null,
    };
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
    loadTiming: loadTimings.loadTiming,
    pipelineLoadTiming: loadTimings.pipelineLoadTiming,
  });
  const decodeBottleneck = buildDecodeBottleneckDiagnostics(metrics, timing);
  const metricsWithTimingDiagnostics = decodeBottleneck
    ? { ...metrics, decodeBottleneck }
    : metrics;
  if (decodeBottleneck) {
    timingDiagnostics.decodeBottleneck = decodeBottleneck;
  }
  const firstLoad = buildFirstLoadComposition({
    modelLoadMs: timing.modelLoadMs,
    firstTokenMs: timing.firstTokenMs,
    firstResponseMs: timing.firstResponseMs,
  });
  const metricsWithContracts = attachExecutionCostLedger(buildSuiteContractMetrics(
    options.suiteName || 'inference',
    loadDiagnostics
      ? { ...metricsWithTimingDiagnostics, load: loadDiagnostics }
      : metricsWithTimingDiagnostics,
    harness.manifest
  ), runtimeConfig, harness.manifest);
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

export function createHarnessPhaseError(error, phase, context = {}) {
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

export async function withHarnessPhase(phase, context, run) {
  try {
    return await run();
  } catch (error) {
    if (error?.details?.harnessPhase) {
      throw error;
    }
    throw createHarnessPhaseError(error, phase, context);
  }
}
