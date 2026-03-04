import assert from 'node:assert/strict';

const {
  createP2PDeliveryObservabilityRecord,
  aggregateP2PDeliveryObservability,
  buildP2PAlertsFromSummary,
  buildP2PDashboardSnapshot,
} = await import('../../src/distribution/p2p-observability.js');

const records = [
  {
    deliveryMetrics: {
      totalDurationMs: 120,
      successSource: 'p2p',
      attemptCount: 1,
      sourceAttempts: { cache: 0, p2p: 1, http: 0 },
      failureCodes: {},
      p2pRttMs: { avg: 12 },
      httpRttMs: { avg: null },
    },
  },
  {
    deliveryMetrics: {
      totalDurationMs: 980,
      successSource: 'http',
      attemptCount: 2,
      sourceAttempts: { cache: 0, p2p: 1, http: 1 },
      failureCodes: {
        DOPPLER_DISTRIBUTION_P2P_TRANSPORT_UNAVAILABLE: 1,
      },
      p2pRttMs: { avg: 0 },
      httpRttMs: { avg: 40 },
    },
  },
];

const record = createP2PDeliveryObservabilityRecord(records[0], {
  modelId: 'test-model',
  shardIndex: 7,
});
assert.equal(record.modelId, 'test-model');
assert.equal(record.shardIndex, 7);
assert.equal(record.p2pHit, true);
assert.equal(record.fallbackToHttp, false);

const summary = aggregateP2PDeliveryObservability(records, {
  targets: {
    minAvailability: 1,
    minP2PHitRate: 0.8,
    maxHttpFallbackRate: 0.1,
    maxP95LatencyMs: 500,
  },
});

assert.equal(summary.totals.records, 2);
assert.equal(summary.totals.p2pHits, 1);
assert.equal(summary.totals.httpFallbacks, 1);
assert.equal(summary.rates.availability, 1);
assert.equal(summary.slo.status, 'fail');
assert.ok(Array.isArray(summary.slo.breaches));
assert.ok(summary.slo.breaches.length >= 2);

const alerts = buildP2PAlertsFromSummary(summary, {
  escalateBreaches: ['http_fallback_rate_breach'],
});
assert.ok(alerts.some((entry) => entry.id === 'http_fallback_rate_breach' && entry.severity === 'critical'));

const snapshot = buildP2PDashboardSnapshot(records);
assert.equal(snapshot.summary.totals.records, 2);
assert.ok(Array.isArray(snapshot.alerts));

console.log('p2p-observability.test: ok');
