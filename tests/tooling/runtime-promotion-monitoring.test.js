import assert from 'node:assert/strict';
import path from 'node:path';

import { computeCanonicalSha256 } from '../../src/utils/canonical-hash.js';
import {
  buildRuntimePromotionMonitoringReport,
  validateRuntimePromotionMonitoringPolicy,
} from '../../tools/check-runtime-promotion-monitoring.js';

const ARTIFACT_ID = `sha256:${'a'.repeat(64)}`;
const EXECUTION_ID = `sha256:${'b'.repeat(64)}`;
const CANDIDATE_HASH = `sha256:${'c'.repeat(64)}`;
const ROLLBACK_HASH = `sha256:${'d'.repeat(64)}`;
const CHANGE_CLASSES = [
  'scheduling-allocation-cache',
  'numerical-kernel',
  'precision-quantization',
  'model-artifact',
  'adapter',
  'provider-integration',
];

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function emptyTargets() {
  return {
    logicalModelIds: [],
    modelIds: [],
    sourceCheckpointIds: [],
    weightPackIds: [],
    manifestVariantIds: [],
    artifactVariantIds: [],
    adapterIds: [],
    adapterDigests: [],
  };
}

function revocations(includeMatch = false) {
  const registry = {
    $schema: 'schema/revocation-registry.schema.json',
    schemaVersion: 1,
    source: 'doppler',
    updatedAtUtc: '2026-08-15T00:00:00.000Z',
    trust: {
      distribution: 'bundled-package',
      signatureVerification: 'unavailable',
    },
    revocations: [],
  };
  if (includeMatch) {
    registry.revocations.push({
      id: 'candidate-regression',
      state: 'revoked',
      issuedAtUtc: '2026-08-15T03:00:00.000Z',
      severity: 'reliability',
      reason: 'Post-promotion regression exceeded the frozen threshold.',
      targets: {
        ...emptyTargets(),
        modelIds: ['model-v1'],
        artifactVariantIds: [ARTIFACT_ID],
      },
      replacements: emptyTargets(),
      evidencePaths: ['docs/goals.md'],
    });
  }
  return registry;
}

function optimizationReceipt() {
  const core = {
    schema: 'doppler.runtime-optimization-receipt/v1',
    candidateId: 'candidate-v1',
    candidateHash: CANDIDATE_HASH,
    campaign: {
      changeClass: 'scheduling-allocation-cache',
    },
    decision: {
      accepted: true,
    },
    promotion: {
      authority: 'human',
      recommended: true,
      runtimeMutationApplied: false,
      requiredStages: ['shadow', 'canary'],
      revocationConditions: [
        'Primary performance degradation or any control or neighboring workload regression.',
      ],
    },
  };
  return { ...core, receiptHash: computeCanonicalSha256(core) };
}

function scope() {
  return {
    modelId: 'model-v1',
    artifactVariantId: ARTIFACT_ID,
    executionId: EXECUTION_ID,
    providerId: 'browser-webgpu',
    environmentFingerprint: 'browser-adapter-driver-v1',
    workloadId: 'decode-p512-d128',
  };
}

function observation(id, value, observedAtUtc, passed = true) {
  return {
    schema: 'doppler.runtime-promotion-observation-evidence/v1',
    id,
    observedAtUtc,
    scope: scope(),
    primaryMetric: {
      id: 'decodeTokensPerSec',
      value,
    },
    controls: [
      { id: 'exact-output', passed },
    ],
    neighbors: [
      { id: 'decode-p256-d128', passed },
    ],
  };
}

function evidenceReference(evidencePath, receipt) {
  return {
    path: evidencePath,
    digest: computeCanonicalSha256(receipt),
  };
}

function policy(status = 'monitoring') {
  const receipt = optimizationReceipt();
  const observationReceipts = status === 'monitoring'
    ? [observation('observation-1', 99, '2026-08-15T01:00:00.000Z')]
    : status === 'retain'
      ? [
        observation('observation-1', 99, '2026-08-15T01:00:00.000Z'),
        observation('observation-2', 101, '2026-08-15T02:00:00.000Z'),
      ]
      : [observation('observation-1', 90, '2026-08-15T01:00:00.000Z')];
  const evidence = new Map();
  const addEvidence = (evidencePath, value) => {
    evidence.set(evidencePath, value);
    return evidenceReference(evidencePath, value);
  };
  const observations = observationReceipts.map((value, index) => addEvidence(
    `evidence/observation-${index + 1}.json`,
    value
  ));
  const rollbackEvidence = addEvidence('evidence/rollback.json', {
    knownSafeTarget: 'baseline-profile-v1',
    verified: true,
  });
  const decisionEvidence = status === 'monitoring'
    ? []
    : [addEvidence('evidence/decision.json', { status, reviewedBy: 'doppler-release' })];
  return {
    receipt,
    evidence,
    value: {
      $schema: '../../src/config/schema/runtime-promotion-monitoring.schema.json',
      schemaVersion: 2,
      source: 'doppler',
      requiredChangeClasses: CHANGE_CLASSES,
      promotions: [
        {
          id: 'candidate-v1-promotion',
          optimizationReceiptPath: 'artifacts/optimization-receipts/candidate-v1.json',
          optimizationReceiptHash: receipt.receiptHash,
          candidateId: 'candidate-v1',
          candidateHash: CANDIDATE_HASH,
          changeClass: 'scheduling-allocation-cache',
          activatedAtUtc: '2026-08-15T00:30:00.000Z',
          scope: scope(),
          rollbackTarget: {
            kind: 'runtime-profile',
            id: 'baseline-profile-v1',
            digest: ROLLBACK_HASH,
            knownSafe: true,
            evidence: rollbackEvidence,
          },
          plan: {
            owner: 'doppler-runtime',
            declaredAtUtc: '2026-08-15T00:00:00.000Z',
            primaryMetric: {
              id: 'decodeTokensPerSec',
              direction: 'maximize',
              baseline: 100,
              maxDegradationPercent: 5,
            },
            controlMetricIds: ['exact-output'],
            neighborWorkloadIds: ['decode-p256-d128'],
            minimumObservations: 2,
            revocationConditions: receipt.promotion.revocationConditions,
          },
          observations,
          decision: status === 'monitoring'
            ? {
              status: 'monitoring',
              decidedAtUtc: null,
              reason: null,
              revocationRecordId: null,
              authority: 'human',
              runtimeMutationApplied: false,
              evidencePaths: [],
            }
            : {
              status,
              decidedAtUtc: '2026-08-15T03:00:00.000Z',
              reason: status === 'retain'
                ? 'Frozen monitoring gates passed.'
                : 'Frozen degradation threshold failed.',
              revocationRecordId: status === 'revoke' ? 'candidate-regression' : null,
              authority: 'human',
              runtimeMutationApplied: false,
              evidencePaths: decisionEvidence,
            },
        },
      ],
    },
  };
}

async function validate(fixture, revocationRegistry = revocations(false)) {
  return validateRuntimePromotionMonitoringPolicy(fixture.value, {
    repoRoot: process.cwd(),
    revocationRegistry,
    loadJson: async (filePath) => {
      const relativePath = path.relative(process.cwd(), filePath);
      if (relativePath === 'artifacts/optimization-receipts/candidate-v1.json') {
        return fixture.receipt;
      }
      if (fixture.evidence.has(relativePath)) return fixture.evidence.get(relativePath);
      throw new Error(`Missing fixture evidence ${relativePath}`);
    },
    pathExists: async () => true,
  });
}

{
  const report = await buildRuntimePromotionMonitoringReport();
  assert.equal(report.ok, true, report.errors.join('\n'));
  assert.equal(report.promotions, 0);
  assert.equal(report.coverageSatisfied, false);
}

{
  const fixture = policy('monitoring');
  const report = await validate(fixture);
  assert.deepEqual(report.errors, []);
  assert.equal(report.monitoring, 1);
  assert.equal(report.coverageSatisfied, false);
}

{
  const fixture = policy('retain');
  const report = await validate(fixture);
  assert.deepEqual(report.errors, []);
  assert.equal(report.retained, 1);
  assert.equal(report.coverageSatisfied, true);
}

{
  const fixture = policy('revoke');
  const report = await validate(fixture, revocations(true));
  assert.deepEqual(report.errors, []);
  assert.equal(report.revoked, 1);
  assert.equal(report.coverageSatisfied, true);
}

{
  const fixture = policy('revoke');
  const report = await validate(fixture);
  assert.ok(
    report.errors.includes('promotion[0]: revoke decision requires a matching active revocation record'),
    report.errors.join('\n')
  );
}

{
  const fixture = policy('retain');
  const reference = fixture.value.promotions[0].observations[1];
  const receipt = fixture.evidence.get(reference.path);
  receipt.scope.providerId = 'node-webgpu';
  reference.digest = computeCanonicalSha256(receipt);
  const report = await validate(fixture);
  assert.ok(report.errors.some((error) => error.includes('scope does not match promotion scope')));
}

{
  const fixture = policy('retain');
  const reference = fixture.value.promotions[0].observations[1];
  fixture.evidence.get(reference.path).primaryMetric.value = 102;
  const report = await validate(fixture);
  assert.ok(
    report.errors.some((error) => error.includes('digest does not match canonical JSON evidence'))
  );
}

{
  const fixture = policy('retain');
  fixture.value.promotions[0].plan.declaredAtUtc = '2026-08-15T04:00:00.000Z';
  const report = await validate(fixture);
  assert.ok(report.errors.includes('promotion[0]: monitor plan must be declared before activation'));
}

{
  const fixture = policy('retain');
  fixture.value.promotions[0].decision.status = 'monitoring';
  const report = await validate(fixture);
  assert.ok(report.errors.includes('promotion[0].decision.status must be retain'));
}

{
  const fixture = policy('retain');
  fixture.value.promotions[0].rollbackTarget.knownSafe = false;
  const report = await validate(fixture);
  assert.ok(report.errors.includes('promotion[0].rollbackTarget.knownSafe must be true'));
}

console.log('runtime-promotion-monitoring.test: ok');
