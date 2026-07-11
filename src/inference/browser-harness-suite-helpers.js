export function buildSuiteSummary(suiteName, results, startTimeMs) {
  let passed = 0;
  let failed = 0;
  let skipped = 0;
  const safeResults = Array.isArray(results) ? results : [];
  for (const result of safeResults) {
    if (result.skipped) {
      skipped++;
    } else if (result.passed) {
      passed++;
    } else {
      failed++;
    }
  }
  const duration = Math.max(0, performance.now() - (Number.isFinite(startTimeMs) ? startTimeMs : performance.now()));
  return { suite: suiteName, passed, failed, skipped, duration, results: safeResults };
}

export function normalizeCacheMode(value) {
  return value === 'cold' || value === 'warm' ? value : 'warm';
}

export function normalizeLoadMode(value, hasModelUrl, modelUrl) {
  if (value === 'opfs' || value === 'http' || value === 'memory' || value === 'file') {
    return value;
  }
  if (!hasModelUrl) return 'opfs';
  if (typeof modelUrl === 'string' && modelUrl.startsWith('file://')) return 'file';
  return 'http';
}

export function normalizeWorkloadType(value) {
  const normalized = String(value || '').trim().toLowerCase();
  return normalized || null;
}

export function safeStatsValue(value) {
  return Number.isFinite(value) ? Number(value) : 0;
}

function calculateRatePerSecond(count, durationMs) {
  const safeCount = safeStatsValue(count);
  const safeDurationMs = safeStatsValue(durationMs);
  if (safeCount <= 0 || safeDurationMs <= 0) return 0;
  return Number(((safeCount * 1000) / safeDurationMs).toFixed(2));
}

export function buildDiffusionPerformanceArtifact({
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
  modality = 'image',
}) {
  const cpuPrefillMs = safeStatsValue(cpuStats?.prefillMs?.median);
  const cpuDenoiseMs = safeStatsValue(cpuStats?.denoiseMs?.median);
  const cpuVaeMs = safeStatsValue(cpuStats?.vaeMs?.median);
  const cpuTotalMs = safeStatsValue(cpuStats?.totalMs?.median);
  const gpuPrefillMs = safeStatsValue(gpuStats?.prefillMs?.median);
  const gpuDenoiseMs = safeStatsValue(gpuStats?.denoiseMs?.median);
  const gpuVaeMs = safeStatsValue(gpuStats?.vaeMs?.median);
  const gpuTotalMs = safeStatsValue(gpuStats?.totalMs?.median);
  const decodeStepsPerSec = calculateRatePerSecond(steps, cpuDenoiseMs);
  const decodeTokensPerSec = calculateRatePerSecond(avgDecodeTokens, cpuDenoiseMs);
  const prefillTokensPerSec = calculateRatePerSecond(avgPrefillTokens, cpuPrefillMs);

  return {
    schemaVersion: 1,
    warmupRuns,
    timedRuns,
    modality,
    shape: {
      width,
      height,
    },
    scheduler: {
      steps,
      guidanceScale,
    },
    cpu: {
      totalMs: cpuTotalMs,
      prefillMs: cpuPrefillMs,
      denoiseMs: cpuDenoiseMs,
      vaeMs: cpuVaeMs,
    },
    gpu: {
      available: gpuStats?.available === true,
      totalMs: gpuStats?.available === true ? gpuTotalMs : null,
      prefillMs: gpuStats?.available === true ? gpuPrefillMs : null,
      denoiseMs: gpuStats?.available === true ? gpuDenoiseMs : null,
      vaeMs: gpuStats?.available === true ? gpuVaeMs : null,
    },
    throughput: {
      prefillTokensPerSec,
      decodeTokensPerSec,
      decodeStepsPerSec,
    },
    tokens: {
      avgPrefillTokens: safeStatsValue(avgPrefillTokens),
      avgDecodeTokens: safeStatsValue(avgDecodeTokens),
    },
  };
}

export function assertDiffusionPerformanceArtifact(metrics, contextLabel = 'diffusion') {
  const artifact = metrics?.performanceArtifact;
  if (!artifact || typeof artifact !== 'object') {
    throw new Error(`${contextLabel}: metrics.performanceArtifact is required.`);
  }
  if (artifact.schemaVersion !== 1) {
    throw new Error(`${contextLabel}: metrics.performanceArtifact.schemaVersion must be 1.`);
  }
  if (!Number.isInteger(artifact.warmupRuns) || artifact.warmupRuns < 0) {
    throw new Error(`${contextLabel}: metrics.performanceArtifact.warmupRuns must be a non-negative integer.`);
  }
  if (!Number.isInteger(artifact.timedRuns) || artifact.timedRuns < 1) {
    throw new Error(`${contextLabel}: metrics.performanceArtifact.timedRuns must be a positive integer.`);
  }
  if (!Number.isFinite(artifact?.cpu?.prefillMs)) {
    throw new Error(`${contextLabel}: metrics.performanceArtifact.cpu.prefillMs must be finite.`);
  }
  if (!Number.isFinite(artifact?.cpu?.denoiseMs)) {
    throw new Error(`${contextLabel}: metrics.performanceArtifact.cpu.denoiseMs must be finite.`);
  }
  if (!Number.isFinite(artifact?.cpu?.vaeMs)) {
    throw new Error(`${contextLabel}: metrics.performanceArtifact.cpu.vaeMs must be finite.`);
  }
  if (!Number.isFinite(artifact?.cpu?.totalMs)) {
    throw new Error(`${contextLabel}: metrics.performanceArtifact.cpu.totalMs must be finite.`);
  }
  if (!Number.isFinite(artifact?.throughput?.decodeStepsPerSec)) {
    throw new Error(`${contextLabel}: metrics.performanceArtifact.throughput.decodeStepsPerSec must be finite.`);
  }
}

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

function metricNumber(value) {
  if (Number.isFinite(value)) {
    return Number(value);
  }
  if (
    value
    && typeof value === 'object'
    && (
      (Number.isFinite(value.samplesAfterOutlierRemoval) && value.samplesAfterOutlierRemoval <= 0)
      || (Number.isFinite(value.samples) && value.samples <= 0)
    )
  ) {
    return null;
  }
  if (value && typeof value === 'object' && Number.isFinite(value.median)) {
    return Number(value.median);
  }
  return null;
}

function shareOfDecode(value, decodeWallMs) {
  if (!Number.isFinite(value) || !Number.isFinite(decodeWallMs) || decodeWallMs <= 0) {
    return null;
  }
  return Number((value / decodeWallMs).toFixed(4));
}

function nullableTimingNumber(value) {
  return Number.isFinite(value) ? toTimingNumber(value, null) : null;
}

function bottleneckClass(componentId) {
  if (componentId === 'command_record') {
    return 'command-record';
  }
  if (
    componentId === 'submit_readback_wait'
    || componentId === 'submit_readback_slack'
    || componentId === 'readback_map_wait'
    || componentId === 'readback_cleanup'
    || componentId === 'readback_copy'
    || componentId === 'submit_readback_unattributed'
  ) {
    return 'submit-readback-wait';
  }
  if (componentId === 'gpu_compute') {
    return 'gpu-compute';
  }
  if (componentId === 'orchestration') {
    return 'orchestration';
  }
  if (componentId === 'unattributed') {
    return 'unattributed';
  }
  return null;
}

function normalizeTopOps(entries) {
  if (!Array.isArray(entries)) {
    return [];
  }
  return entries
    .slice(0, 8)
    .map((entry) => ({
      label: typeof entry?.label === 'string' && entry.label.length > 0 ? entry.label : null,
      count: Number.isFinite(entry?.count) ? Number(entry.count) : null,
      shareOfOps: Number.isFinite(entry?.shareOfOps) ? Number(entry.shareOfOps) : null,
    }))
    .filter((entry) => entry.label || entry.count != null || entry.shareOfOps != null);
}

function normalizeUniformCacheForDiagnostics(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }
  const normalized = {};
  for (const key of [
    'hits',
    'misses',
    'totalLookups',
    'hitRateRatio',
    'evictions',
    'currentSize',
    'pendingDestruction',
  ]) {
    if (Number.isFinite(value[key])) {
      normalized[key] = Number(value[key]);
    }
  }
  if (typeof value.hitRate === 'string' && value.hitRate.length > 0) {
    normalized.hitRate = value.hitRate;
  }
  return Object.keys(normalized).length > 0 ? normalized : null;
}

export function buildDecodeBottleneckDiagnostics(metrics = {}, timing = {}) {
  const gpu = metrics?.gpu && typeof metrics.gpu === 'object' ? metrics.gpu : {};
  const decodeWallMs = metricNumber(metrics?.decodeMs)
    ?? metricNumber(metrics?.latency?.decodeMs)
    ?? metricNumber(timing?.decodeMs);
  if (!Number.isFinite(decodeWallMs) || decodeWallMs <= 0) {
    return null;
  }

  const commandRecordMs = metricNumber(gpu.decodeRecordMs);
  const commandRecordOps = metricNumber(gpu.decodeRecordOps);
  const commandRecordPasses = metricNumber(gpu.decodeRecordPasses);
  const commandRecordMsPerOp = metricNumber(gpu.decodeRecordMsPerOp);
  const commandRecordMsPerPass = metricNumber(gpu.decodeRecordMsPerPass);
  const commandRecordPassesPerOp = metricNumber(gpu.decodeRecordPassesPerOp);
  const commandRecordMsPerExecutedBatchToken = metricNumber(gpu.decodeRecordMsPerExecutedBatchToken);
  const commandRecordOpsPerExecutedBatchToken = metricNumber(gpu.decodeRecordOpsPerExecutedBatchToken);
  const commandRecordPassesPerExecutedBatchToken = metricNumber(gpu.decodeRecordPassesPerExecutedBatchToken);
  const commandRecordUniqueOpLabels = metricNumber(gpu.decodeRecordUniqueOpLabels);
  const submitWaitMs = metricNumber(gpu.decodeSubmitWaitMs);
  const readbackWaitMs = metricNumber(gpu.decodeReadbackWaitMs);
  const readbackMapWaitMs = metricNumber(gpu.decodeReadbackMapWaitMs);
  const readbackCleanupMs = metricNumber(gpu.decodeReadbackCleanupMs);
  const readbackCopyMs = metricNumber(gpu.decodeReadbackCopyMs);
  const orchestrationMs = metricNumber(gpu.decodeOrchestrationMs);
  const gpuTimestampMs = metricNumber(gpu.decodeMs);
  const uniformCache = normalizeUniformCacheForDiagnostics(gpu.uniformCache);
  const waitCandidates = [submitWaitMs, readbackWaitMs].filter(Number.isFinite);
  const effectiveSubmitReadbackWaitMs = waitCandidates.length > 0 ? Math.max(...waitCandidates) : null;
  const submitReadbackSlackMs = (
    Number.isFinite(effectiveSubmitReadbackWaitMs)
    && Number.isFinite(gpuTimestampMs)
  )
    ? Math.max(0, effectiveSubmitReadbackWaitMs - gpuTimestampMs)
    : null;
  const readbackSubcomponentMs = [
    readbackMapWaitMs,
    readbackCleanupMs,
    readbackCopyMs,
  ]
    .filter(Number.isFinite)
    .reduce((sum, value) => sum + value, 0);
  const hasReadbackSubcomponents = readbackSubcomponentMs > 0;
  const readbackUnattributedMs = (
    hasReadbackSubcomponents
    && Number.isFinite(effectiveSubmitReadbackWaitMs)
  )
    ? Math.max(0, effectiveSubmitReadbackWaitMs - readbackSubcomponentMs)
    : null;
  const accountedMs = [
    commandRecordMs,
    effectiveSubmitReadbackWaitMs,
    orchestrationMs,
  ]
    .filter(Number.isFinite)
    .reduce((sum, value) => sum + value, 0);
  const residualMs = Math.max(0, decodeWallMs - accountedMs);
  const components = [
    { id: 'command_record', label: 'command recording', ms: commandRecordMs },
    {
      id: Number.isFinite(gpuTimestampMs) ? 'submit_readback_slack' : 'submit_readback_wait',
      label: Number.isFinite(gpuTimestampMs) ? 'submit/readback slack' : 'submit/readback wait',
      ms: hasReadbackSubcomponents
        ? null
        : (Number.isFinite(gpuTimestampMs) ? submitReadbackSlackMs : effectiveSubmitReadbackWaitMs),
    },
    { id: 'readback_map_wait', label: 'readback map wait', ms: readbackMapWaitMs },
    { id: 'readback_cleanup', label: 'readback cleanup', ms: readbackCleanupMs },
    { id: 'readback_copy', label: 'readback CPU copy', ms: readbackCopyMs },
    { id: 'submit_readback_unattributed', label: 'submit/readback unattributed', ms: readbackUnattributedMs },
    { id: 'gpu_compute', label: 'GPU timestamp work', ms: gpuTimestampMs },
    { id: 'orchestration', label: 'decode orchestration', ms: orchestrationMs },
    { id: 'unattributed', label: 'unattributed decode wall', ms: residualMs },
  ].filter((component) => Number.isFinite(component.ms) && component.ms > 0);
  const dominant = components.length > 0
    ? components.reduce((best, component) => component.ms > best.ms ? component : best, components[0])
    : null;
  const dominantRecord = dominant
    ? {
      id: dominant.id,
      label: dominant.label,
      ms: nullableTimingNumber(dominant.ms),
      shareOfDecode: shareOfDecode(dominant.ms, decodeWallMs),
    }
    : null;

  return {
    schemaVersion: 1,
    source: 'doppler',
    dominant: dominantRecord,
    bottleneckClass: dominantRecord ? bottleneckClass(dominantRecord.id) : null,
    decodeWallMs: nullableTimingNumber(decodeWallMs),
    componentsMs: {
      commandRecordMs: nullableTimingNumber(commandRecordMs),
      submitWaitMs: nullableTimingNumber(submitWaitMs),
      readbackWaitMs: nullableTimingNumber(readbackWaitMs),
      effectiveSubmitReadbackWaitMs: nullableTimingNumber(effectiveSubmitReadbackWaitMs),
      readbackMapWaitMs: nullableTimingNumber(readbackMapWaitMs),
      readbackCleanupMs: nullableTimingNumber(readbackCleanupMs),
      readbackCopyMs: nullableTimingNumber(readbackCopyMs),
      readbackUnattributedMs: nullableTimingNumber(readbackUnattributedMs),
      gpuTimestampMs: nullableTimingNumber(gpuTimestampMs),
      submitReadbackSlackMs: nullableTimingNumber(submitReadbackSlackMs),
      orchestrationMs: nullableTimingNumber(orchestrationMs),
      residualMs: nullableTimingNumber(residualMs),
    },
    recording: {
      opCount: nullableTimingNumber(commandRecordOps),
      passCount: nullableTimingNumber(commandRecordPasses),
      uniqueOpLabels: nullableTimingNumber(commandRecordUniqueOpLabels),
      msPerOp: nullableTimingNumber(commandRecordMsPerOp),
      msPerPass: nullableTimingNumber(commandRecordMsPerPass),
      passesPerOp: nullableTimingNumber(commandRecordPassesPerOp),
      msPerExecutedBatchToken: nullableTimingNumber(commandRecordMsPerExecutedBatchToken),
      opsPerExecutedBatchToken: nullableTimingNumber(commandRecordOpsPerExecutedBatchToken),
      passesPerExecutedBatchToken: nullableTimingNumber(commandRecordPassesPerExecutedBatchToken),
      topOps: normalizeTopOps(gpu.decodeRecordTopOps),
      topOpGroups: normalizeTopOps(gpu.decodeRecordTopOpGroups),
      uniformCache,
    },
    shares: {
      commandRecord: shareOfDecode(commandRecordMs, decodeWallMs),
      submitWait: shareOfDecode(submitWaitMs, decodeWallMs),
      readbackWait: shareOfDecode(readbackWaitMs, decodeWallMs),
      effectiveSubmitReadbackWait: shareOfDecode(effectiveSubmitReadbackWaitMs, decodeWallMs),
      readbackMapWait: shareOfDecode(readbackMapWaitMs, decodeWallMs),
      readbackCleanup: shareOfDecode(readbackCleanupMs, decodeWallMs),
      readbackCopy: shareOfDecode(readbackCopyMs, decodeWallMs),
      readbackUnattributed: shareOfDecode(readbackUnattributedMs, decodeWallMs),
      gpuTimestamp: shareOfDecode(gpuTimestampMs, decodeWallMs),
      submitReadbackSlack: shareOfDecode(submitReadbackSlackMs, decodeWallMs),
      orchestration: shareOfDecode(orchestrationMs, decodeWallMs),
      residual: shareOfDecode(residualMs, decodeWallMs),
    },
    semantics: {
      effectiveSubmitReadbackWaitMs: 'max(decodeSubmitWaitMs, decodeReadbackWaitMs); submit and readback waits overlap.',
      readbackMapWaitMs: 'Wall time awaiting staging-buffer mapAsync; usually includes GPU completion behind the readback.',
      gpuTimestampMs: 'Timestamp-query GPU work when available; null means it was not captured.',
      topOps: 'Highest-count exact compute-pass labels observed during command recording.',
      topOpGroups: 'Highest-count compute-pass labels after grouping repeated per-layer labels.',
      uniformCache: 'Current uniform-buffer cache counters at the end of the measured suite.',
    },
  };
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
