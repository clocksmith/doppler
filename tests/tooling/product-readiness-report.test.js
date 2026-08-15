import assert from 'node:assert/strict';

import { buildProductReadinessReport } from '../../tools/render-product-readiness-report.js';

const report = await buildProductReadinessReport();
const revocations = report.contracts.revocations;

assert.equal(report.ok, true);
assert.equal(report.actions.length, 9);
assert.equal(report.actions[0].code, 'maintained-application-integrations-missing');
assert.equal(report.actions[0].owner, 'doppler-product');
assert.equal(report.actions[0].completionClass, 'application');
assert.equal(report.actions.at(-1).code, 'signed-live-revocation-authority-missing');
assert.equal(report.actions.at(-1).completionClass, 'production-authority');
assert.equal(report.contracts.productIntegrations.qualified, 0);
assert.equal(report.contracts.productIntegrations.candidates, 3);
assert.deepEqual(report.contracts.productIntegrations.candidateWorkloads, [
  'generation',
  'embedding-retrieval',
  'reranking',
]);
assert.deepEqual(
  report.contracts.productIntegrations.candidateDetails.map((entry) => entry.id),
  [
    'reploid-local-generation',
    'dream-local-embedding-retrieval',
    'columbo-local-reranking',
  ]
);
assert.equal(revocations.ok, true);
assert.equal(revocations.active, revocations.bundled.active);
assert.equal(revocations.signatureVerification, revocations.bundled.signatureVerification);
assert.equal(revocations.bundled.signatureVerification, 'unavailable');
assert.equal(revocations.signedLive.mechanismAvailable, true);
assert.equal(revocations.signedLive.authorityQualified, false);
assert.equal(revocations.signedLive.schema, 'doppler.signed-revocation-envelope/v1');
assert.equal(revocations.signedLive.signatureAlgorithm, 'ECDSA-P256-SHA256');
assert.equal(revocations.signedLive.configuration, 'explicit-application');
assert.equal(revocations.signedLive.backgroundRefresh, false);

console.log('product-readiness-report.test: ok');
