import assert from 'node:assert/strict';

import {
  assertComparableFingerprints,
  buildComparisonFingerprint,
  listObservationPolicies,
} from '../../src/client/inspection.js';

const policies = listObservationPolicies();
assert.deepEqual(
  policies.map((policy) => policy.id),
  ['demo/always-on', 'demo/guided-quality', 'demo/deep-xray']
);
assert.equal(
  policies.find((policy) => policy.id === 'demo/always-on').gpuTimestampQueries,
  false
);
assert.equal(
  policies.find((policy) => policy.id === 'demo/deep-xray').allowedClaimTypes.includes('diagnostic'),
  true
);

const base = {
  artifact: {
    modelId: 'toy-model',
    manifestHash: `sha256:${'a'.repeat(64)}`,
  },
  tokenizer: { type: 'bpe', revision: 'one' },
  promptTokenIds: [1, 2],
  sampling: { temperature: 0 },
  observationPolicyId: 'demo/always-on',
  execution: { backend: 'webgpu', executionPlanId: 'plan-one' },
  browser: { userAgent: 'browser-one' },
  adapter: { vendor: 'vendor-one' },
};
const first = buildComparisonFingerprint(base);
const changedPlan = buildComparisonFingerprint({
  ...base,
  execution: { backend: 'webgpu', executionPlanId: 'plan-two' },
});
assert.throws(
  () => assertComparableFingerprints('performance', first, changedPlan),
  /performance fingerprints differ/
);

console.log('diagnostics-controller-contract.test: ok');
