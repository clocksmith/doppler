import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { projectPhysicalSummary, buildPhysicalAcceptanceReport } from '../../tools/product-readiness-evidence.js';
import { buildBunProductQualificationReport } from '../../tools/check-bun-product-qualification.js';

import {
  buildProductReadinessReport,
  buildProductReadinessState,
  formatProductReadinessMarkdown,
} from '../../tools/render-product-readiness-report.js';

const report = await buildProductReadinessReport();
const revocations = report.contracts.revocations;

assert.equal(report.ok, true);
assert.equal(report.readiness.contractValid, true);
assert.equal(report.schema, 'doppler.product-readiness/v2');
assert.equal(report.readiness.portfolioGatesSatisfied, false);
assert.equal(report.readiness.adoption.independentAdoptionProven, false);
assert.equal(report.readiness.deployment.broaderGoalProven, false);
assert.ok(report.readiness.adoption.blockers.includes('external-executable-model-adoption-missing'));
assert.equal(report.readiness.networkProven, false);
assert.equal(report.readiness.technical.anyConfigurationProven, true);
assert.equal(report.readiness.technical.checkoutQualified, false);
assert.equal(report.readiness.technical.packageSelection, 'retained-acceptance-archive');
assert.ok(report.readiness.deployment.blockers.includes('customer-electron-fleet-receipts-missing'));
assert.equal(report.readiness.support.contractValid, true);
assert.ok(report.readiness.support.declarations.every((entry) => entry.owner && entry.tier));
for (const field of ['productReady', 'localHardwareProven', 'standaloneProven', 'technicalAcceptance']) {
  assert.equal(Object.hasOwn(report.readiness, field), false, `ambiguous ${field} removed in report v2`);
}
assert.equal(report.contracts.productIntegrations.gateSatisfied, true);
assert.equal(report.contracts.providerConformance.ok, true);
assert.equal(report.contracts.providerConformance.gateSatisfied, false);

const markdown = formatProductReadinessMarkdown(report);
assert.match(markdown, /^## Readiness$/mu);
assert.match(markdown, /^- contract valid: yes$/mu);
assert.match(markdown, /^- portfolio gates satisfied: no$/mu);
assert.match(markdown, /^- retained configuration execution proven: yes \(only rows below\)$/mu);
assert.match(markdown, /^- independent adoption proven: no$/mu);
assert.doesNotMatch(markdown, /^- product ready:/mu);
assert.match(markdown, /^  - `external-executable-model-adoption-missing`$/mu);
assert.doesNotMatch(markdown, /^- status: ok$/mu);

const invalidContractReadiness = buildProductReadinessState({
  goals: {
    goals: [
      {
        id: 'local-webgpu-product-surface',
        claimAllowed: true,
        blockers: [],
      },
      {
        id: 'correctness-performance-claims',
        claimableRows: 1,
      },
    ],
  },
  productIntegrations: { gateSatisfied: true },
  providerConformance: { gateSatisfied: true },
}, false);
assert.equal(invalidContractReadiness.deployment.broaderGoalProven, true);
assert.equal(invalidContractReadiness.contractValid, false);
assert.equal(invalidContractReadiness.technical.anyConfigurationProven, false, 'generic claimable rows are not physical GPU evidence');
// Pure projection fixtures, not promoted adoption evidence. A network success
// cannot substitute for adoption, and missing commercial evidence cannot block it.
const projection = {
  goals: { goals: [
    { id: 'open-execution-network', acceptanceScope: 'technical-network', status: 'complete', claimAllowed: true, blockers: [] },
    { id: 'local-webgpu-product-surface', acceptanceScope: 'standalone', claimAllowed: false,
      blockers: ['paid-doppler-production-release-missing'], rowStates: [
        { id: 'external-executable-model-adoption', status: 'partial', claimAllowed: false,
          blockers: ['external-executable-model-adoption-missing'] },
      ] },
  ] },
  productIntegrations: { gateSatisfied: false }, providerConformance: { gateSatisfied: false },
};
assert.equal(buildProductReadinessState(projection, true).technical.anyConfigurationProven, false);
assert.equal(buildProductReadinessState(projection, false).technical.anyConfigurationProven, false);
projection.goals.goals[0].claimAllowed = false;
projection.goals.goals[0].status = 'partial';
projection.goals.goals[1].claimAllowed = true;
assert.equal(buildProductReadinessState(projection, true).technical.anyConfigurationProven, false);
projection.goals.goals[1].claimAllowed = false;
const adoption = projection.goals.goals[1].rowStates[0];
Object.assign(adoption, { status: 'covered', claimAllowed: true, blockers: [] });
assert.equal(buildProductReadinessState(projection, true).adoption.independentAdoptionProven, true);
assert.equal(buildProductReadinessState(projection, true).technical.anyConfigurationProven, false, 'adoption cannot manufacture physical proof');
assert.equal(buildProductReadinessState(projection, true).networkProven, false);
assert.equal(buildProductReadinessState(projection, true).deployment.broaderGoalProven, false);
assert.equal(buildProductReadinessState(projection, false).contractValid, false);
adoption.blockers = ['external-executable-model-adoption-missing'];
projection.physicalAcceptance = report.readiness.technical;
assert.equal(buildProductReadinessState(projection, true).technical.anyConfigurationProven, true, 'missing adopters cannot erase physical qualification');
assert.equal(buildProductReadinessState(projection, true).adoption.independentAdoptionProven, false);
assert.equal(report.adoptionActions.length, 1);
assert.equal(report.adoptionActions[0].code, 'external-executable-model-adoption-missing');
assert.equal(report.adoptionActions[0].owner, 'doppler-product');
assert.equal(report.adoptionActions[0].completionClass, 'application');

const physical = JSON.parse(await readFile(new URL('../../artifacts/bounded-consolidation-2026-09-20/contracts-embedding-reranking-summary.json', import.meta.url), 'utf8'));
const project = (summary, digest = physical.package.sha256) => projectPhysicalSummary(summary, { evidencePath: 'fixture', packageSha256: digest });
assert.ok(project(physical).configurations.every((entry) => entry.physicalExecutionProven));
for (const mutate of [
  (entry) => { entry.passed = false; },
  (entry) => { entry.cleanupErrors.push('cleanup failed'); },
  (entry) => { entry.results[0].hardware.isFallbackAdapter = true; },
  (entry) => { entry.results[0].targetPlanDigest = null; },
  (entry) => { entry.results[0].comparisons[0].checks[0].passed = false; },
  (entry) => { entry.results[0].checks = []; },
  (entry) => { entry.results[0].observations = []; },
]) {
  const fixture = structuredClone(physical);
  mutate(fixture);
  assert.equal(project(fixture).configurations[0].physicalExecutionProven, false);
}
const malformed = structuredClone(physical);
malformed.originalReceiptSha256 = null;
assert.equal(project(malformed).errors.length, 1);
const differentArchive = await buildPhysicalAcceptanceReport({ packageSha256: '0'.repeat(64) });
assert.equal(differentArchive.ok, true, 'a valid report can honestly lack matching execution');
assert.ok(differentArchive.configurations.every((entry) => !entry.physicalExecutionProven && entry.blockers.includes('package-identity-mismatch')));
assert.equal((await buildPhysicalAcceptanceReport({ packageSha256: '0.6.2' })).ok, false, 'version is not archive identity');
assert.equal((await buildPhysicalAcceptanceReport({ acceptanceRecord: 'missing-acceptance.json' })).ok, false);
assert.equal(
  report.supportingActions.find((action) => action.code === 'signed-live-revocation-authority-missing')?.completionClass,
  'production-authority'
);
assert.equal(report.contracts.productIntegrations.qualified, 3);
assert.equal(report.contracts.electronProspects.ok, true);
assert.equal(report.contracts.electronProspects.researched, 5);
assert.equal(report.contracts.electronProspects.primary, 3);
assert.equal(report.contracts.electronProspects.qualifiedCustomers, 0);
assert.deepEqual(
  report.contracts.electronProspects.orderedTargets.map((entry) => entry.id),
  ['anythingllm', 'joplin', 'cherry-studio', 'chatbox', 'affine']
);
assert.ok(report.contracts.electronProspects.orderedTargets.every(
  (entry) => entry.claimAllowed === false
));
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

// Projection only: the real registry is unchanged, and still has no qualified Bun workload.
const bunFixture = await buildBunProductQualificationReport();
bunFixture.qualifications[2] = { ...bunFixture.qualifications[2], qualified: true, claimAllowed: true, reasons: [], blockers: [], missingEvidence: [] };
bunFixture.qualifiedWorkloads = 1;
const partialBun = await buildProductReadinessReport({ bunQualificationBuilder: async () => bunFixture });
assert.equal(partialBun.contracts.bunQualification.qualifiedDetails[0].workload, 'reranking');
assert.equal(partialBun.contracts.bunQualification.portfolioQualified, false);
assert.equal(partialBun.contracts.bunQualification.subsystemTier, 'experimental');
assert.match(formatProductReadinessMarkdown(partialBun), /qwen3-reranking-bun-product: reranking/);

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
