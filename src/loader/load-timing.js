export function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

export function roundLoadTimingMs(value) {
  return Number.isFinite(value) ? Number(value.toFixed(3)) : null;
}

export function createLoadTiming(modelId, hasCustomLoader) {
  return {
    schemaVersion: 1,
    source: 'doppler-loader',
    modelId: typeof modelId === 'string' ? modelId : null,
    status: 'running',
    customShardLoader: hasCustomLoader === true,
    byteAccountingMode: hasCustomLoader === true
      ? 'custom-loader-read-progress'
      : 'full-shard-progress',
    totalBytes: null,
    totalShards: null,
    bytesLoaded: 0,
    shardsLoaded: 0,
    bytesPerSecond: null,
    phasesMs: {
      preflight: null,
      tensorLocations: null,
      embeddings: null,
      layers: null,
      finalWeights: null,
      cleanup: null,
    },
    layers: {
      count: null,
      totalMs: null,
      meanMs: null,
      maxMs: null,
      maxLayer: null,
    },
    totalMs: null,
    failedPhase: null,
    error: null,
  };
}

export function finishLoadPhase(loadTiming, phase, startMs) {
  if (!loadTiming?.phasesMs || !phase) return;
  loadTiming.phasesMs[phase] = roundLoadTimingMs(nowMs() - startMs);
}

export function finishLoadTiming(loadTiming, status, startMs, error = null, failedPhase = null) {
  if (!loadTiming) return;
  const totalMs = nowMs() - startMs;
  loadTiming.status = status;
  loadTiming.totalMs = roundLoadTimingMs(totalMs);
  loadTiming.bytesPerSecond = totalMs > 0 && Number.isFinite(loadTiming.bytesLoaded)
    ? Math.round((loadTiming.bytesLoaded / totalMs) * 1000)
    : null;
  if (status === 'failed') {
    loadTiming.failedPhase = failedPhase;
    loadTiming.error = error?.message ?? String(error ?? 'unknown load error');
  }
}
