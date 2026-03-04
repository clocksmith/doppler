export const P2P_OBSERVABILITY_SCHEMA_VERSION = 1;

const DEFAULT_SLO_TARGETS = Object.freeze({
  minAvailability: 0.995,
  minP2PHitRate: 0.5,
  maxHttpFallbackRate: 0.5,
  maxP95LatencyMs: 5000,
});

function asFiniteNumber(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function asNonNegativeInteger(value, fallback = 0) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) {
    return fallback;
  }
  return parsed;
}

function asStringOrNull(value) {
  if (value === undefined || value === null) {
    return null;
  }
  const normalized = String(value).trim();
  return normalized || null;
}

function sumFailureCodes(failureCodes) {
  if (!failureCodes || typeof failureCodes !== 'object') {
    return 0;
  }
  let total = 0;
  for (const value of Object.values(failureCodes)) {
    total += asNonNegativeInteger(value, 0);
  }
  return total;
}

function mergeFailureCodes(into, failureCodes) {
  if (!failureCodes || typeof failureCodes !== 'object') {
    return;
  }
  for (const [code, rawCount] of Object.entries(failureCodes)) {
    const key = asStringOrNull(code) ?? 'unknown';
    const count = asNonNegativeInteger(rawCount, 0);
    into[key] = (into[key] ?? 0) + count;
  }
}

function percentile(values, ratio) {
  if (!Array.isArray(values) || values.length === 0) {
    return null;
  }
  const sorted = values
    .filter((entry) => Number.isFinite(entry))
    .sort((left, right) => left - right);
  if (sorted.length === 0) {
    return null;
  }
  if (sorted.length === 1) {
    return sorted[0];
  }
  const index = Math.ceil(sorted.length * ratio) - 1;
  const clampedIndex = Math.max(0, Math.min(sorted.length - 1, index));
  return sorted[clampedIndex];
}

function resolveSLOTargets(options = {}) {
  const targets = options.targets && typeof options.targets === 'object'
    ? options.targets
    : {};

  return {
    minAvailability: asFiniteNumber(targets.minAvailability, DEFAULT_SLO_TARGETS.minAvailability),
    minP2PHitRate: asFiniteNumber(targets.minP2PHitRate, DEFAULT_SLO_TARGETS.minP2PHitRate),
    maxHttpFallbackRate: asFiniteNumber(targets.maxHttpFallbackRate, DEFAULT_SLO_TARGETS.maxHttpFallbackRate),
    maxP95LatencyMs: asFiniteNumber(targets.maxP95LatencyMs, DEFAULT_SLO_TARGETS.maxP95LatencyMs),
  };
}

function normalizeDeliveryMetrics(input) {
  if (!input || typeof input !== 'object') {
    throw new Error('P2P observability requires an object payload.');
  }
  const metrics = input.deliveryMetrics && typeof input.deliveryMetrics === 'object'
    ? input.deliveryMetrics
    : input;
  if (!metrics || typeof metrics !== 'object') {
    throw new Error('P2P observability metrics payload is missing deliveryMetrics.');
  }
  return metrics;
}

export function createP2PDeliveryObservabilityRecord(input, context = {}) {
  const metrics = normalizeDeliveryMetrics(input);
  const sourceAttempts = metrics.sourceAttempts && typeof metrics.sourceAttempts === 'object'
    ? metrics.sourceAttempts
    : {};
  const p2pAttempts = asNonNegativeInteger(sourceAttempts.p2p, 0);
  const httpAttempts = asNonNegativeInteger(sourceAttempts.http, 0);
  const cacheAttempts = asNonNegativeInteger(sourceAttempts.cache, 0);
  const successSource = asStringOrNull(metrics.successSource);
  const attemptCount = asNonNegativeInteger(metrics.attemptCount, p2pAttempts + httpAttempts + cacheAttempts);
  const totalDurationMs = asFiniteNumber(metrics.totalDurationMs, 0);
  const p2pRttAvgMs = Number.isFinite(metrics?.p2pRttMs?.avg) ? metrics.p2pRttMs.avg : null;
  const httpRttAvgMs = Number.isFinite(metrics?.httpRttMs?.avg) ? metrics.httpRttMs.avg : null;

  return {
    schemaVersion: P2P_OBSERVABILITY_SCHEMA_VERSION,
    timestampMs: asNonNegativeInteger(context.timestampMs ?? Date.now(), Date.now()),
    modelId: asStringOrNull(context.modelId),
    shardIndex: Number.isInteger(context.shardIndex) ? context.shardIndex : null,
    successSource,
    attemptCount,
    p2pAttempts,
    httpAttempts,
    cacheAttempts,
    totalDurationMs,
    p2pRttAvgMs,
    httpRttAvgMs,
    totalFailures: sumFailureCodes(metrics.failureCodes),
    fallbackToHttp: successSource === 'http' && p2pAttempts > 0,
    p2pHit: successSource === 'p2p',
    failureCodes: metrics.failureCodes && typeof metrics.failureCodes === 'object'
      ? { ...metrics.failureCodes }
      : {},
    rawMetrics: metrics,
  };
}

export function aggregateP2PDeliveryObservability(records = [], options = {}) {
  if (!Array.isArray(records)) {
    throw new Error('P2P observability aggregate expects records array.');
  }

  const normalized = records.map((record) => createP2PDeliveryObservabilityRecord(record));
  const total = normalized.length;
  const successful = normalized.filter((record) => !!record.successSource).length;
  const p2pHits = normalized.filter((record) => record.p2pHit).length;
  const httpFallbacks = normalized.filter((record) => record.fallbackToHttp).length;
  const durationValues = normalized.map((record) => record.totalDurationMs);

  const failureCodes = {};
  for (const record of normalized) {
    mergeFailureCodes(failureCodes, record.failureCodes);
  }

  const targets = resolveSLOTargets(options);
  const availability = total > 0 ? successful / total : 0;
  const p2pHitRate = total > 0 ? p2pHits / total : 0;
  const httpFallbackRate = total > 0 ? httpFallbacks / total : 0;
  const p95LatencyMs = percentile(durationValues, 0.95);

  const breaches = [];
  if (availability < targets.minAvailability) {
    breaches.push({
      id: 'availability_breach',
      metric: 'availability',
      expected: `>= ${targets.minAvailability}`,
      actual: availability,
    });
  }
  if (p2pHitRate < targets.minP2PHitRate) {
    breaches.push({
      id: 'p2p_hit_rate_breach',
      metric: 'p2pHitRate',
      expected: `>= ${targets.minP2PHitRate}`,
      actual: p2pHitRate,
    });
  }
  if (httpFallbackRate > targets.maxHttpFallbackRate) {
    breaches.push({
      id: 'http_fallback_rate_breach',
      metric: 'httpFallbackRate',
      expected: `<= ${targets.maxHttpFallbackRate}`,
      actual: httpFallbackRate,
    });
  }
  if (p95LatencyMs != null && p95LatencyMs > targets.maxP95LatencyMs) {
    breaches.push({
      id: 'latency_p95_breach',
      metric: 'p95LatencyMs',
      expected: `<= ${targets.maxP95LatencyMs}`,
      actual: p95LatencyMs,
    });
  }

  return {
    schemaVersion: P2P_OBSERVABILITY_SCHEMA_VERSION,
    generatedAtMs: Date.now(),
    totals: {
      records: total,
      successful,
      failed: Math.max(0, total - successful),
      p2pHits,
      httpFallbacks,
    },
    rates: {
      availability,
      p2pHitRate,
      httpFallbackRate,
    },
    latencyMs: {
      p50: percentile(durationValues, 0.5),
      p95: p95LatencyMs,
      p99: percentile(durationValues, 0.99),
    },
    failureCodes,
    slo: {
      targets,
      breaches,
      status: breaches.length === 0 ? 'pass' : 'fail',
    },
  };
}

export function buildP2PAlertsFromSummary(summary, options = {}) {
  if (!summary || typeof summary !== 'object') {
    throw new Error('P2P observability alert builder requires summary object.');
  }

  const severityByBreach = {
    availability_breach: 'critical',
    p2p_hit_rate_breach: 'warning',
    http_fallback_rate_breach: 'warning',
    latency_p95_breach: 'warning',
  };
  const escalatedBreaches = new Set(Array.isArray(options.escalateBreaches)
    ? options.escalateBreaches.map((entry) => asStringOrNull(entry)).filter(Boolean)
    : []);

  const breaches = Array.isArray(summary?.slo?.breaches)
    ? summary.slo.breaches
    : [];

  return breaches.map((breach) => {
    const id = asStringOrNull(breach.id) ?? 'unknown_breach';
    const baseSeverity = severityByBreach[id] ?? 'warning';
    const severity = escalatedBreaches.has(id)
      ? 'critical'
      : baseSeverity;
    return {
      schemaVersion: P2P_OBSERVABILITY_SCHEMA_VERSION,
      id,
      severity,
      message: `P2P SLO breach (${id}): expected ${breach.expected}, actual ${breach.actual}.`,
      metric: breach.metric ?? null,
      expected: breach.expected ?? null,
      actual: breach.actual ?? null,
      generatedAtMs: Date.now(),
    };
  });
}

export function buildP2PDashboardSnapshot(records = [], options = {}) {
  const summary = aggregateP2PDeliveryObservability(records, options);
  const alerts = buildP2PAlertsFromSummary(summary, options);
  return {
    schemaVersion: P2P_OBSERVABILITY_SCHEMA_VERSION,
    generatedAtMs: Date.now(),
    summary,
    alerts,
  };
}
