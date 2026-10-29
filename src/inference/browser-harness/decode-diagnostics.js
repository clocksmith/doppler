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
  return Number.isFinite(value) ? Math.round(value * 1000) / 1000 : null;
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
  // Aggregate durations do not carry aligned intervals or timestamp coverage.
  // Subtracting GPU work from a completion wait cannot recover host-only time.
  const submitReadbackSlackMs = null;
  const readbackUnattributedMs = null;
  const residualMs = null;
  const hasReadbackSubcomponents = [readbackMapWaitMs, readbackCleanupMs, readbackCopyMs]
    .some((value) => Number.isFinite(value) && value > 0);
  const components = [
    { id: 'command_record', label: 'command recording', ms: commandRecordMs },
    {
      id: 'submit_readback_wait',
      label: 'submit/readback wait',
      ms: hasReadbackSubcomponents
        ? null
        : effectiveSubmitReadbackWaitMs,
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
    schemaVersion: 2,
    source: 'doppler',
    dominant: dominantRecord,
    bottleneckClass: null,
    attribution: {
      status: 'unresolved',
      reason: 'Aggregate intervals may overlap and do not establish a critical path or causal bottleneck.',
    },
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
      dominant: 'Largest observed aggregate interval, not a causal bottleneck or recoverable fraction of decode time.',
      shares: 'Each interval divided by decode wall; intervals overlap and shares are not additive.',
      effectiveSubmitReadbackWaitMs: 'Maximum aggregate submit/readback wait, not a measured union of intervals.',
      submitReadbackSlackMs: 'Unknown without aligned GPU/host intervals and complete timestamp coverage.',
      readbackUnattributedMs: 'Unknown: independently aggregated subcomponent medians are not subtractable.',
      residualMs: 'Unknown: overlapping aggregate intervals cannot partition decode wall.',
      readbackMapWaitMs: 'Wall time awaiting staging-buffer mapAsync; usually includes GPU completion behind the readback.',
      gpuTimestampMs: 'Elapsed timestamped GPU intervals when available; may include scheduling/preemption, omit unmeasured passes, and overlap host recording. Null means not captured.',
      topOps: 'Highest-count exact compute-pass labels observed during command recording.',
      topOpGroups: 'Highest-count compute-pass labels after grouping repeated per-layer labels.',
      uniformCache: 'Current uniform-buffer cache counters at the end of the measured suite.',
    },
  };
}
