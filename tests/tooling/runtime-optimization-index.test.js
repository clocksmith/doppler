import assert from 'node:assert/strict';
import { buildRuntimeOptimizationResultsIndex } from '../../src/tooling/runtime-optimization-index.js';

function receipt(status, reason, suffix) {
  return {
    schema: 'doppler.runtime-optimization-receipt/v1',
    candidateKind: 'registered-kernel-variant',
    campaign: {
      changeClass: 'numerical-kernel',
      retryConditions: ['kernel identity changes'],
      revocationConditions: ['output parity regresses'],
    },
    registeredReference: { digest: `sha256:${suffix.repeat(64)}` },
    candidateHash: `sha256:${suffix.repeat(64)}`,
    model: { modelId: 'test-model' },
    runtimeInputs: { candidate: { runtimeConfig: {} } },
    measurement: {
      metricPath: 'result.metrics.decodeTokensPerSec',
      pairs: [{
        valid: true,
        candidate: { deviceInfo: { vendor: 'test' } },
      }],
    },
    decision: {
      accepted: status === 'accepted',
      status,
      reasons: reason ? [reason] : [],
    },
    receiptHash: `sha256:${suffix.repeat(64)}`,
  };
}

const index = buildRuntimeOptimizationResultsIndex([
  receipt('rejected', 'neighboring_workload_guard_failed', '1'),
  receipt('invalid', 'baseline_verification_failed', '1'),
  receipt('accepted', null, '2'),
]);
assert.equal(index.schema, 'doppler.runtime-optimization-results-index/v1');
assert.equal(index.receiptCount, 3);
assert.equal(index.negativeResultCount, 2);
assert.equal(index.entries.length, 2);
const negativeEntry = index.entries.find(
  (entry) => entry.identity.candidateReferenceDigest.endsWith('1'.repeat(64))
);
assert.equal(negativeEntry.reasons.neighboring_workload_guard_failed, 1);
assert.equal(negativeEntry.retryConditions['kernel identity changes'], 2);
assert.equal(negativeEntry.revocationConditions['output parity regresses'], 2);

console.log('runtime-optimization-index.test: ok');
