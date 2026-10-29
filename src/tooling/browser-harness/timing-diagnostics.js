import { normalizeCacheMode } from './suite-summary.js';

function formatMetricNumber(value, fallback = 0, digits = 2) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) return fallback;
  return Number(numericValue.toFixed(digits));
}

export function toTimingNumber(value, fallback = 0) {
  return formatMetricNumber(value, fallback, 2);
}

export function safeToFixed(value, fallback = 0, digits = 2) {
  return formatMetricNumber(value, fallback, digits);
}

export function sampleTimingNumber(stats, key, fallback = 0) {
  return formatMetricNumber(stats?.[key], fallback, 2);
}

export function buildCanonicalTiming(overrides = {}) {
  const cacheMode = normalizeCacheMode(overrides.cacheMode);
  const modelLoadMs = toTimingNumber(overrides.modelLoadMs, 0);
  const prefillMs = toTimingNumber(overrides.prefillMs, 0);
  const decodeMs = toTimingNumber(overrides.decodeMs, 0);
  const decodeMsPerTokenP50 = Number.isFinite(overrides.decodeMsPerTokenP50)
    ? toTimingNumber(overrides.decodeMsPerTokenP50)
    : null;
  const decodeMsPerTokenP95 = Number.isFinite(overrides.decodeMsPerTokenP95)
    ? toTimingNumber(overrides.decodeMsPerTokenP95)
    : null;
  const decodeMsPerTokenP99 = Number.isFinite(overrides.decodeMsPerTokenP99)
    ? toTimingNumber(overrides.decodeMsPerTokenP99)
    : null;
  const decodeTokensPerSec = Number.isFinite(overrides.decodeTokensPerSec)
    ? toTimingNumber(overrides.decodeTokensPerSec)
    : null;
  const prefillTokensPerSec = Number.isFinite(overrides.prefillTokensPerSec)
    ? toTimingNumber(overrides.prefillTokensPerSec)
    : null;
  const totalRunMs = toTimingNumber(
    overrides.totalRunMs,
    toTimingNumber(prefillMs + decodeMs)
  );
  const firstTokenMs = Number.isFinite(overrides.firstTokenMs)
    ? toTimingNumber(overrides.firstTokenMs)
    : null;
  const firstResponseMs = Number.isFinite(overrides.firstResponseMs)
    ? toTimingNumber(overrides.firstResponseMs)
    : toTimingNumber(modelLoadMs + totalRunMs);

  return {
    modelLoadMs,
    firstTokenMs,
    firstResponseMs,
    prefillMs,
    decodeMs,
    decodeMsPerTokenP50,
    decodeMsPerTokenP95,
    decodeMsPerTokenP99,
    decodeTokensPerSec,
    prefillTokensPerSec,
    totalRunMs,
    cacheMode,
    loadMode: overrides.loadMode,
  };
}

function normalizeLoadTimingInteger(value) {
  return Number.isFinite(value) ? Math.max(0, Math.round(value)) : null;
}

function normalizeLoadTimingMs(value) {
  return Number.isFinite(value) ? toTimingNumber(value, null) : null;
}

function normalizeLoadTimingString(value) {
  return typeof value === 'string' && value.length > 0 ? value : null;
}

function normalizeLoaderLoadTiming(loadTiming) {
  if (!loadTiming || typeof loadTiming !== 'object' || Array.isArray(loadTiming)) {
    return null;
  }
  const rawPhasesMs = loadTiming.phasesMs && typeof loadTiming.phasesMs === 'object'
    ? loadTiming.phasesMs
    : {};
  const phasesMs = {
    preflight: normalizeLoadTimingMs(rawPhasesMs.preflight),
    tensorLocations: normalizeLoadTimingMs(rawPhasesMs.tensorLocations),
    embeddings: normalizeLoadTimingMs(rawPhasesMs.embeddings),
    layers: normalizeLoadTimingMs(rawPhasesMs.layers),
    finalWeights: normalizeLoadTimingMs(rawPhasesMs.finalWeights),
    cleanup: normalizeLoadTimingMs(rawPhasesMs.cleanup),
  };
  const rawLayers = loadTiming.layers && typeof loadTiming.layers === 'object'
    ? loadTiming.layers
    : {};
  const status = typeof loadTiming.status === 'string' && loadTiming.status.length > 0
    ? loadTiming.status
    : null;
  return {
    schemaVersion: Number.isInteger(loadTiming.schemaVersion) ? loadTiming.schemaVersion : 1,
    source: typeof loadTiming.source === 'string' && loadTiming.source.length > 0
      ? loadTiming.source
      : 'doppler-loader',
    modelId: typeof loadTiming.modelId === 'string' ? loadTiming.modelId : null,
    status,
    customShardLoader: loadTiming.customShardLoader === true,
    byteAccountingMode: typeof loadTiming.byteAccountingMode === 'string' && loadTiming.byteAccountingMode.length > 0
      ? loadTiming.byteAccountingMode
      : (loadTiming.customShardLoader === true ? 'custom-loader-read-progress' : 'full-shard-progress'),
    totalBytes: normalizeLoadTimingInteger(loadTiming.totalBytes),
    totalShards: normalizeLoadTimingInteger(loadTiming.totalShards),
    bytesLoaded: normalizeLoadTimingInteger(loadTiming.bytesLoaded),
    shardsLoaded: normalizeLoadTimingInteger(loadTiming.shardsLoaded),
    bytesPerSecond: normalizeLoadTimingInteger(loadTiming.bytesPerSecond),
    phasesMs,
    layers: {
      count: normalizeLoadTimingInteger(rawLayers.count),
      totalMs: normalizeLoadTimingMs(rawLayers.totalMs),
      meanMs: normalizeLoadTimingMs(rawLayers.meanMs),
      maxMs: normalizeLoadTimingMs(rawLayers.maxMs),
      maxLayer: normalizeLoadTimingInteger(rawLayers.maxLayer),
    },
    totalMs: normalizeLoadTimingMs(loadTiming.totalMs),
    failedPhase: typeof loadTiming.failedPhase === 'string' ? loadTiming.failedPhase : null,
    error: typeof loadTiming.error === 'string' ? loadTiming.error : null,
  };
}

function normalizeTokenizerLoadTiming(tokenizerLoadTiming) {
  if (!tokenizerLoadTiming || typeof tokenizerLoadTiming !== 'object' || Array.isArray(tokenizerLoadTiming)) {
    return null;
  }
  const rawPhasesMs = tokenizerLoadTiming.phasesMs && typeof tokenizerLoadTiming.phasesMs === 'object'
    ? tokenizerLoadTiming.phasesMs
    : {};
  return {
    schemaVersion: Number.isInteger(tokenizerLoadTiming.schemaVersion) ? tokenizerLoadTiming.schemaVersion : 1,
    source: normalizeLoadTimingString(tokenizerLoadTiming.source) ?? 'doppler-tokenizer',
    modelId: typeof tokenizerLoadTiming.modelId === 'string' ? tokenizerLoadTiming.modelId : null,
    status: normalizeLoadTimingString(tokenizerLoadTiming.status),
    tokenizerType: normalizeLoadTimingString(tokenizerLoadTiming.tokenizerType),
    tokenizerFile: normalizeLoadTimingString(tokenizerLoadTiming.tokenizerFile),
    backend: normalizeLoadTimingString(tokenizerLoadTiming.backend),
    assetSource: normalizeLoadTimingString(tokenizerLoadTiming.assetSource),
    cacheHit: tokenizerLoadTiming.cacheHit === true,
    phasesMs: {
      configResolution: normalizeLoadTimingMs(rawPhasesMs.configResolution),
      cacheLookup: normalizeLoadTimingMs(rawPhasesMs.cacheLookup),
      backendCreate: normalizeLoadTimingMs(rawPhasesMs.backendCreate),
      assetLoad: normalizeLoadTimingMs(rawPhasesMs.assetLoad),
      assetParse: normalizeLoadTimingMs(rawPhasesMs.assetParse),
      backendLoad: normalizeLoadTimingMs(rawPhasesMs.backendLoad),
      cacheStore: normalizeLoadTimingMs(rawPhasesMs.cacheStore),
    },
    totalMs: normalizeLoadTimingMs(tokenizerLoadTiming.totalMs),
    error: typeof tokenizerLoadTiming.error === 'string' ? tokenizerLoadTiming.error : null,
  };
}

function normalizePipelineLoadTiming(pipelineLoadTiming) {
  if (!pipelineLoadTiming || typeof pipelineLoadTiming !== 'object' || Array.isArray(pipelineLoadTiming)) {
    return null;
  }
  const rawPhasesMs = pipelineLoadTiming.phasesMs && typeof pipelineLoadTiming.phasesMs === 'object'
    ? pipelineLoadTiming.phasesMs
    : {};
  const phasesMs = {
    reset: normalizeLoadTimingMs(rawPhasesMs.reset),
    configResolution: normalizeLoadTimingMs(rawPhasesMs.configResolution),
    kernelWarmup: normalizeLoadTimingMs(rawPhasesMs.kernelWarmup),
    tokenizer: normalizeLoadTimingMs(rawPhasesMs.tokenizer),
    executionSetup: normalizeLoadTimingMs(rawPhasesMs.executionSetup),
    loadWeights: normalizeLoadTimingMs(rawPhasesMs.loadWeights),
    rope: normalizeLoadTimingMs(rawPhasesMs.rope),
    convStates: normalizeLoadTimingMs(rawPhasesMs.convStates),
  };
  return {
    schemaVersion: Number.isInteger(pipelineLoadTiming.schemaVersion) ? pipelineLoadTiming.schemaVersion : 1,
    source: typeof pipelineLoadTiming.source === 'string' && pipelineLoadTiming.source.length > 0
      ? pipelineLoadTiming.source
      : 'doppler-pipeline',
    modelId: typeof pipelineLoadTiming.modelId === 'string' ? pipelineLoadTiming.modelId : null,
    status: typeof pipelineLoadTiming.status === 'string' && pipelineLoadTiming.status.length > 0
      ? pipelineLoadTiming.status
      : null,
    phasesMs,
    details: {
      tokenizer: normalizeTokenizerLoadTiming(pipelineLoadTiming.details?.tokenizer),
    },
    totalMs: normalizeLoadTimingMs(pipelineLoadTiming.totalMs),
  };
}

export function buildLoadTimingDiagnostics(modelLoadMs, loadTiming, pipelineLoadTiming = null) {
  const loader = normalizeLoaderLoadTiming(loadTiming);
  const pipeline = normalizePipelineLoadTiming(pipelineLoadTiming);
  if (!loader && !pipeline) {
    return null;
  }
  const normalizedModelLoadMs = Number.isFinite(modelLoadMs)
    ? toTimingNumber(modelLoadMs, null)
    : null;
  const modelLoadMinusLoaderMs = (
    Number.isFinite(normalizedModelLoadMs)
    && Number.isFinite(loader?.totalMs)
  )
    ? toTimingNumber(normalizedModelLoadMs - loader.totalMs, null)
    : null;
  const modelLoadMinusPipelineMs = (
    Number.isFinite(normalizedModelLoadMs)
    && Number.isFinite(pipeline?.totalMs)
  )
    ? toTimingNumber(normalizedModelLoadMs - pipeline.totalMs, null)
    : null;
  const pipelineMinusLoaderMs = (
    Number.isFinite(pipeline?.totalMs)
    && Number.isFinite(loader?.totalMs)
  )
    ? toTimingNumber(pipeline.totalMs - loader.totalMs, null)
    : null;
  return {
    schemaVersion: 1,
    source: 'doppler',
    semantics: {
      modelLoadMs: 'suite initialization time before generation',
      loaderTotalMs: 'DopplerLoader.load() weight-loader wall time',
      pipelineTotalMs: 'InferencePipeline.loadModel() wall time',
      modelLoadMinusPipelineMs: 'modelLoadMs - pipeline.totalMs; harness, storage, manifest, GPU init, and pipeline construction',
      pipelineMinusLoaderMs: 'pipeline.totalMs - loader.totalMs; config, tokenizer, KV, RoPE, and other pipeline setup',
      modelLoadMinusLoaderMs: 'modelLoadMs - loader.totalMs',
    },
    modelLoadMs: normalizedModelLoadMs,
    loader,
    pipeline,
    residualsMs: {
      modelLoadMinusLoaderMs,
      modelLoadMinusPipelineMs,
      pipelineMinusLoaderMs,
    },
    consistent: {
      loaderWithinModelLoad: Number.isFinite(modelLoadMinusLoaderMs)
        ? modelLoadMinusLoaderMs >= -2
        : null,
      pipelineWithinModelLoad: Number.isFinite(modelLoadMinusPipelineMs)
        ? modelLoadMinusPipelineMs >= -2
        : null,
      loaderWithinPipeline: Number.isFinite(pipelineMinusLoaderMs)
        ? pipelineMinusLoaderMs >= -2
        : null,
    },
  };
}

export function buildTimingDiagnostics(timing = {}, options = {}) {
  const prefillSemantics = String(options.prefillSemantics || 'internal_prefill_phase');
  const decodeSemantics = String(options.decodeSemantics || 'time after first token');
  const source = String(options.source || 'doppler');
  const modelLoadMs = Number.isFinite(timing.modelLoadMs) ? toTimingNumber(timing.modelLoadMs) : null;
  const firstTokenMs = Number.isFinite(timing.firstTokenMs) ? toTimingNumber(timing.firstTokenMs) : null;
  const firstResponseMs = Number.isFinite(timing.firstResponseMs) ? toTimingNumber(timing.firstResponseMs) : null;
  const prefillMs = Number.isFinite(timing.prefillMs) ? toTimingNumber(timing.prefillMs) : null;
  const decodeMs = Number.isFinite(timing.decodeMs) ? toTimingNumber(timing.decodeMs) : null;
  const totalRunMs = Number.isFinite(timing.totalRunMs) ? toTimingNumber(timing.totalRunMs) : null;

  const firstResponseFromLoadAndFirstTokenMs = (
    Number.isFinite(modelLoadMs) && Number.isFinite(firstTokenMs)
  )
    ? toTimingNumber(modelLoadMs + firstTokenMs)
    : null;
  const runFromPrefillAndDecodeMs = (
    Number.isFinite(prefillMs) && Number.isFinite(decodeMs)
  )
    ? toTimingNumber(prefillMs + decodeMs)
    : null;

  const firstResponseResidualMs = (
    Number.isFinite(firstResponseMs) && Number.isFinite(firstResponseFromLoadAndFirstTokenMs)
  )
    ? toTimingNumber(firstResponseMs - firstResponseFromLoadAndFirstTokenMs)
    : null;
  const runResidualMs = (
    Number.isFinite(totalRunMs) && Number.isFinite(runFromPrefillAndDecodeMs)
  )
    ? toTimingNumber(totalRunMs - runFromPrefillAndDecodeMs)
    : null;

  const diagnostics = {
    schemaVersion: 1,
    source,
    semantics: {
      modelLoadMs: 'model initialization/load before generation',
      firstTokenMs: 'ttft from generation start',
      firstResponseMs: 'modelLoadMs + firstTokenMs',
      prefillMs: prefillSemantics,
      decodeMs: decodeSemantics,
      totalRunMs: 'prefillMs + decodeMs',
    },
    componentsMs: {
      modelLoadMs,
      firstTokenMs,
      firstResponseMs,
      prefillMs,
      decodeMs,
      totalRunMs,
    },
    sumsMs: {
      firstResponseFromLoadAndFirstTokenMs,
      runFromPrefillAndDecodeMs,
    },
    residualsMs: {
      firstResponseResidualMs,
      runResidualMs,
    },
    consistent: {
      firstResponse: Number.isFinite(firstResponseResidualMs) ? Math.abs(firstResponseResidualMs) <= 2 : null,
      totalRun: Number.isFinite(runResidualMs) ? Math.abs(runResidualMs) <= 2 : null,
    },
  };
  const load = buildLoadTimingDiagnostics(modelLoadMs, options.loadTiming, options.pipelineLoadTiming);
  if (load) {
    diagnostics.load = load;
  }
  return diagnostics;
}


// Mirrors `buildFirstLoadComposition` on the transformers.js runner
// (benchmarks/runners/transformersjs-bench.js) so Doppler bench receipts expose
// the same six-field first-load breakdown. Fields Doppler does not yet
// instrument (browserLaunchMs, pageReadyMs, cachePrimeMs) are `null` —
// `null` explicitly means "this Doppler surface does not separate that
// phase" (nullable-required-field convention). Sums and residuals fall back
// to `null` whenever any dependency is null.
export function buildFirstLoadComposition(fields = {}) {
  const browserLaunchMs = Number.isFinite(fields.browserLaunchMs)
    ? toTimingNumber(fields.browserLaunchMs)
    : null;
  const pageReadyMs = Number.isFinite(fields.pageReadyMs)
    ? toTimingNumber(fields.pageReadyMs)
    : null;
  const cachePrimeMs = Number.isFinite(fields.cachePrimeMs)
    ? toTimingNumber(fields.cachePrimeMs)
    : null;
  const modelLoadMs = Number.isFinite(fields.modelLoadMs)
    ? toTimingNumber(fields.modelLoadMs)
    : null;
  const firstTokenMs = Number.isFinite(fields.firstTokenMs)
    ? toTimingNumber(fields.firstTokenMs)
    : null;
  const firstResponseMs = Number.isFinite(fields.firstResponseMs)
    ? toTimingNumber(fields.firstResponseMs)
    : null;

  const firstResponseFromLoadAndFirstTokenMs = (
    Number.isFinite(modelLoadMs) && Number.isFinite(firstTokenMs)
  )
    ? toTimingNumber(modelLoadMs + firstTokenMs)
    : null;
  const harnessWarmStartToFirstResponseMs = (
    Number.isFinite(pageReadyMs)
    && Number.isFinite(cachePrimeMs)
    && Number.isFinite(firstResponseMs)
  )
    ? toTimingNumber(pageReadyMs + cachePrimeMs + firstResponseMs)
    : null;
  const endToEndFirstResponseMs = (
    Number.isFinite(browserLaunchMs) && Number.isFinite(harnessWarmStartToFirstResponseMs)
  )
    ? toTimingNumber(browserLaunchMs + harnessWarmStartToFirstResponseMs)
    : null;
  const firstResponseResidualMs = (
    Number.isFinite(firstResponseMs) && Number.isFinite(firstResponseFromLoadAndFirstTokenMs)
  )
    ? toTimingNumber(firstResponseMs - firstResponseFromLoadAndFirstTokenMs)
    : null;

  return {
    schemaVersion: 1,
    semantics: {
      browserLaunchMs: 'node launch request -> browser/context ready',
      pageReadyMs: 'runner navigation + startup',
      cachePrimeMs: 'untimed warm-opfs prefetch/load pass',
      modelLoadMs: 'model initialization/load before generation',
      firstTokenMs: 'ttft from generation start',
      firstResponseMs: 'modelLoadMs + firstTokenMs',
      harnessWarmStartToFirstResponseMs: 'pageReadyMs + cachePrimeMs + firstResponseMs',
      endToEndFirstResponseMs: 'browserLaunchMs + pageReadyMs + cachePrimeMs + firstResponseMs',
    },
    componentsMs: {
      browserLaunchMs,
      pageReadyMs,
      cachePrimeMs,
      modelLoadMs,
      firstTokenMs,
      firstResponseMs,
    },
    sumsMs: {
      firstResponseFromLoadAndFirstTokenMs,
      harnessWarmStartToFirstResponseMs,
      endToEndFirstResponseMs,
    },
    residualsMs: {
      firstResponseResidualMs,
    },
    consistent: {
      firstResponse: Number.isFinite(firstResponseResidualMs)
        ? Math.abs(firstResponseResidualMs) <= 2
        : null,
    },
  };
}
