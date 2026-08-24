import assert from 'node:assert/strict';

import { buildProductReadinessReport } from '../../tools/render-product-readiness-report.js';

const report = await buildProductReadinessReport();
const revocations = report.contracts.revocations;

assert.equal(report.ok, true);
assert.equal(report.actions.length, 9);
assert.equal(report.actions[0].code, 'paid-doppler-production-release-missing');
assert.equal(report.actions[0].owner, 'doppler-product');
assert.equal(report.actions[0].completionClass, 'application');
assert.equal(report.actions.at(-1).code, 'pack-first-surface-migration-incomplete');
assert.equal(report.actions.at(-1).completionClass, 'repository');
assert.equal(
  report.actions.find((action) => action.code === 'signed-live-revocation-authority-missing')?.completionClass,
  'production-authority'
);
assert.equal(report.contracts.productIntegrations.qualified, 3);
assert.equal(report.contracts.productPortfolioCoherence.ok, true);
assert.equal(report.contracts.productPortfolioCoherence.workloads.length, 3);
assert.equal(report.contracts.productPortfolioCoherence.requiredGates.length, 4);
assert.equal(report.contracts.productIntegrations.candidates, 0);
assert.deepEqual(report.contracts.productIntegrations.candidateWorkloads, []);
assert.deepEqual(
  report.contracts.productIntegrations.candidateDetails.map((entry) => entry.id),
  []
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
assert.equal(report.contracts.bunQualification.ok, true);
assert.equal(report.contracts.bunQualification.gateSatisfied, false);
assert.equal(report.contracts.bunQualification.qualified, 0);
assert.equal(report.contracts.bunQualification.candidates, 3);
assert.deepEqual(report.contracts.bunQualification.candidateWorkloads, [
  'generation',
  'embedding',
  'reranking',
]);
assert.deepEqual(
  report.contracts.bunQualification.candidateDetails.map((entry) => entry.id),
  [
    'qwen35-generation-bun-product',
    'embeddinggemma-bun-product',
    'qwen3-reranking-bun-product',
  ]
);
assert.equal(report.contracts.bunQualification.subsystemTier, 'experimental');
assert.equal(report.contracts.bunQualification.releaseEngineStatus, 'experimental');
assert.equal(report.contracts.bunQualification.releaseTargetStatus, 'experimental');

const invalidBun = await buildProductReadinessReport({
  bunQualificationBuilder: async () => ({
    ok: false,
    errors: ['fixture Bun contract failure'],
    gateSatisfied: false,
    qualifiedWorkloads: 0,
    candidateWorkloads: 0,
    qualifications: [],
    missingWorkloads: ['generation', 'embedding', 'reranking'],
    portfolioQualified: false,
    subsystemTier: 'experimental',
    releaseEngineStatus: 'experimental',
    releaseTargetStatus: 'experimental',
  }),
});
assert.equal(invalidBun.ok, false);
assert.ok(invalidBun.errors.includes('Bun qualification: fixture Bun contract failure'));

const invalidPortfolio = await buildProductReadinessReport({
  productPortfolioCoherenceBuilder: async () => ({
    ok: false,
    errors: ['fixture portfolio drift'],
    workloads: [],
    requiredGates: [
      'product-integration',
      'provider-conformance',
      'runtime-ownership',
      'bun-product',
    ],
  }),
});
assert.equal(invalidPortfolio.ok, false);
assert.ok(invalidPortfolio.errors.includes(
  'product portfolio coherence: fixture portfolio drift'
));
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
