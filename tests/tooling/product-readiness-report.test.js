import assert from 'node:assert/strict';

import { buildProductReadinessReport } from '../../tools/render-product-readiness-report.js';

const report = await buildProductReadinessReport();
const revocations = report.contracts.revocations;

assert.equal(report.ok, true);
assert.equal(report.actions.length, 6);
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
assert.equal(report.contracts.providerConformance.qualified, 0);
assert.equal(report.contracts.providerConformance.candidates, 3);
assert.deepEqual(report.contracts.providerConformance.candidateWorkloads, [
  'generation',
  'embedding',
  'reranking',
]);
assert.deepEqual(
  report.contracts.providerConformance.candidateDetails.map((entry) => entry.id),
  [
    'qwen35-generation-browser-node',
    'embeddinggemma-browser-node',
    'qwen3-reranking-browser-node',
  ]
);
assert.equal(report.contracts.runtimeOwnership.qualified, 0);
assert.equal(report.contracts.runtimeOwnership.candidates, 3);
assert.deepEqual(report.contracts.runtimeOwnership.candidateWorkloads, [
  'generation',
  'embedding',
  'reranking',
]);
assert.deepEqual(
  report.contracts.runtimeOwnership.candidateDetails.map((entry) => entry.id),
  [
    'qwen35-generation-runtime-ownership',
    'embeddinggemma-runtime-ownership',
    'qwen3-reranking-runtime-ownership',
  ]
);
assert.equal(revocations.ok, true);
assert.equal(revocations.active, revocations.bundled.active);
assert.equal(revocations.signatureVerification, revocations.bundled.signatureVerification);
assert.equal(revocations.bundled.signatureVerification, 'unavailable');
assert.equal(revocations.signedLive.mechanismAvailable, true);
assert.equal(revocations.signedLive.qualificationContractOk, true);
assert.equal(revocations.signedLive.authorityQualified, false);
assert.equal(revocations.signedLive.qualifiedAuthorities, 0);
assert.equal(revocations.signedLive.candidateAuthorities, 1);
assert.equal(revocations.signedLive.authorityDetails[0].id, 'doppler-production-revocation-authority');
assert.deepEqual(revocations.signedLive.requiredHosts, ['browser', 'node']);
assert.ok(revocations.signedLive.requiredDrills.includes('compromise-recovery'));
assert.equal(revocations.signedLive.schema, 'doppler.signed-revocation-envelope/v1');
assert.equal(revocations.signedLive.signatureAlgorithm, 'ECDSA-P256-SHA256');
assert.equal(revocations.signedLive.configuration, 'explicit-application');
assert.equal(revocations.signedLive.backgroundRefresh, false);

console.log('product-readiness-report.test: ok');
