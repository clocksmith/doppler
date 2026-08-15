import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { computeCanonicalSha256 } from '../../src/utils/canonical-hash.js';
import {
  parseArgs,
  recordRuntimePromotionMonitoring,
} from '../../tools/record-runtime-promotion-monitoring.js';

const SOURCE_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const NOW = new Date('2026-08-16T00:00:00.000Z');
const ARTIFACT_ID = `sha256:${'a'.repeat(64)}`;
const EXECUTION_ID = `sha256:${'b'.repeat(64)}`;
const CANDIDATE_HASH = `sha256:${'c'.repeat(64)}`;
const ROLLBACK_HASH = `sha256:${'d'.repeat(64)}`;
const REVIEWER_REVISION = '1'.repeat(40);

async function writeJson(filePath, value) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
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

function optimizationReceipt() {
  const core = {
    schema: 'doppler.runtime-optimization-receipt/v1',
    candidateId: 'candidate-v1',
    candidateHash: CANDIDATE_HASH,
    campaign: { changeClass: 'scheduling-allocation-cache' },
    decision: { accepted: true },
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

function activationReceipt(overrides = {}) {
  return {
    schema: 'doppler.runtime-promotion-activation-evidence/v1',
    promotionId: 'candidate-v1-promotion',
    candidateId: 'candidate-v1',
    candidateHash: CANDIDATE_HASH,
    activatedAtUtc: '2026-08-15T00:30:00.000Z',
    scope: scope(),
    authority: 'human',
    reviewer: 'doppler-release',
    reviewerRevision: REVIEWER_REVISION,
    statement: 'The reviewer activated this exact candidate and monitoring scope.',
    ...overrides,
  };
}

function observation(id, value, observedAtUtc, passed = true) {
  return {
    schema: 'doppler.runtime-promotion-observation-evidence/v1',
    id,
    observedAtUtc,
    scope: scope(),
    primaryMetric: { id: 'decodeTokensPerSec', value },
    controls: [{ id: 'exact-output', passed }],
    neighbors: [{ id: 'decode-p256-d128', passed }],
  };
}

function decisionReceipt(status, overrides = {}) {
  return {
    schema: 'doppler.runtime-promotion-decision-evidence/v1',
    promotionId: 'candidate-v1-promotion',
    candidateId: 'candidate-v1',
    candidateHash: CANDIDATE_HASH,
    scope: scope(),
    status,
    decidedAtUtc: '2026-08-15T03:00:00.000Z',
    reason: status === 'retain'
      ? 'Frozen monitoring gates passed.'
      : 'Frozen degradation threshold failed.',
    revocationRecordId: status === 'revoke' ? 'candidate-regression' : null,
    authority: 'human',
    reviewer: 'doppler-release',
    reviewerRevision: REVIEWER_REVISION,
    statement: 'The reviewer accepted the evaluator-derived terminal outcome.',
    ...overrides,
  };
}

function capture(status) {
  const terminal = status !== 'monitoring';
  return {
    schema: 'doppler.runtime-promotion-monitoring-capture/v1',
    promotionId: 'candidate-v1-promotion',
    optimizationReceiptPath: 'evidence/optimization.json',
    activationEvidencePath: 'evidence/activation.json',
    rollbackTarget: {
      kind: 'runtime-profile',
      id: 'baseline-profile-v1',
      digest: ROLLBACK_HASH,
      evidencePath: 'evidence/rollback.json',
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
    },
    observationPaths: terminal
      ? ['evidence/observation-1.json', 'evidence/observation-2.json']
      : ['evidence/observation-1.json'],
    decisionEvidencePath: terminal ? 'evidence/decision.json' : null,
  };
}

async function createFixture(status = 'monitoring') {
  const repoRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-monitor-record-'));
  const policy = JSON.parse(await fs.readFile(
    path.join(SOURCE_ROOT, 'tools/policies/runtime-promotion-monitoring.json'),
    'utf8'
  ));
  const revocations = JSON.parse(await fs.readFile(
    path.join(SOURCE_ROOT, 'src/config/revocation-registry.json'),
    'utf8'
  ));
  const policyPath = path.join(repoRoot, 'tools/policies/runtime-promotion-monitoring.json');
  const revocationPath = path.join(repoRoot, 'src/config/revocation-registry.json');
  const capturePath = path.join(repoRoot, 'capture.json');
  const outputPolicyPath = path.join(repoRoot, 'monitoring.next.json');
  await writeJson(policyPath, policy);
  await writeJson(revocationPath, revocations);
  await writeJson(path.join(repoRoot, 'evidence/optimization.json'), optimizationReceipt());
  await writeJson(path.join(repoRoot, 'evidence/activation.json'), activationReceipt());
  await writeJson(path.join(repoRoot, 'evidence/rollback.json'), {
    knownSafeTarget: 'baseline-profile-v1',
    verified: true,
  });
  await writeJson(
    path.join(repoRoot, 'evidence/observation-1.json'),
    observation('observation-1', 99, '2026-08-15T01:00:00.000Z')
  );
  if (status !== 'monitoring') {
    const revoke = status === 'revoke';
    await writeJson(
      path.join(repoRoot, 'evidence/observation-2.json'),
      observation('observation-2', revoke ? 90 : 101, '2026-08-15T02:00:00.000Z')
    );
    await writeJson(path.join(repoRoot, 'evidence/decision.json'), decisionReceipt(status));
  }
  await writeJson(capturePath, capture(status));
  return { repoRoot, policyPath, revocationPath, capturePath, outputPolicyPath };
}

async function withFixture(status, run) {
  const fixture = await createFixture(status);
  try {
    await run(fixture);
  } finally {
    await fs.rm(fixture.repoRoot, { recursive: true, force: true });
  }
}

await withFixture('monitoring', async (fixture) => {
  const result = await recordRuntimePromotionMonitoring({ ...fixture, now: NOW });
  assert.equal(result.status, 'monitoring');
  assert.equal(result.coverageSatisfied, false);
  assert.equal(result.runtimeMutationApplied, false);
  const output = JSON.parse(await fs.readFile(fixture.outputPolicyPath, 'utf8'));
  const promotion = output.promotions[0];
  assert.equal(promotion.candidateId, 'candidate-v1');
  assert.equal(promotion.candidateHash, CANDIDATE_HASH);
  assert.deepEqual(promotion.scope, scope());
  assert.equal(promotion.decision.evidence, null);
  assert.match(promotion.activationEvidence.digest, /^sha256:[0-9a-f]{64}$/);
});

await withFixture('retain', async (fixture) => {
  const result = await recordRuntimePromotionMonitoring({ ...fixture, now: NOW });
  assert.equal(result.status, 'retain');
  assert.equal(result.coverageSatisfied, true);
  const output = JSON.parse(await fs.readFile(fixture.outputPolicyPath, 'utf8'));
  assert.equal(output.promotions[0].decision.authority, 'human');
  assert.match(output.promotions[0].decision.evidence.digest, /^sha256:[0-9a-f]{64}$/);
});

await withFixture('monitoring', async (fixture) => {
  await writeJson(
    path.join(fixture.repoRoot, 'evidence/activation.json'),
    activationReceipt({ authority: 'automation' })
  );
  await assert.rejects(
    () => recordRuntimePromotionMonitoring({ ...fixture, now: NOW }),
    /Activation evidence is invalid.*authority must be human/
  );
});

await withFixture('retain', async (fixture) => {
  await writeJson(
    path.join(fixture.repoRoot, 'evidence/decision.json'),
    decisionReceipt('revoke')
  );
  await assert.rejects(
    () => recordRuntimePromotionMonitoring({ ...fixture, now: NOW }),
    /Recorded promotion monitoring policy is invalid.*decision.status must be retain/
  );
});

await withFixture('monitoring', async (fixture) => {
  await recordRuntimePromotionMonitoring({ ...fixture, now: NOW });
  await fs.copyFile(fixture.outputPolicyPath, fixture.policyPath);
  await assert.rejects(
    () => recordRuntimePromotionMonitoring({ ...fixture, now: NOW }),
    /already exists/
  );
  const result = await recordRuntimePromotionMonitoring({ ...fixture, now: NOW, replace: true });
  assert.equal(result.status, 'monitoring');
});

await withFixture('monitoring', async (fixture) => {
  await recordRuntimePromotionMonitoring({ ...fixture, now: NOW });
  await fs.copyFile(fixture.outputPolicyPath, fixture.policyPath);
  const captureValue = capture('monitoring');
  captureValue.observationPaths = [];
  await writeJson(fixture.capturePath, captureValue);
  await assert.rejects(
    () => recordRuntimePromotionMonitoring({ ...fixture, now: NOW, replace: true }),
    /must retain all prior monitoring observations/
  );
});

await withFixture('retain', async (fixture) => {
  await recordRuntimePromotionMonitoring({ ...fixture, now: NOW });
  await fs.copyFile(fixture.outputPolicyPath, fixture.policyPath);
  await assert.rejects(
    () => recordRuntimePromotionMonitoring({ ...fixture, now: NOW, replace: true }),
    /Terminal promotion candidate-v1-promotion cannot be replaced/
  );
});

assert.throws(
  () => parseArgs([
    '--capture',
    'capture.json',
    '--out',
    'tools/policies/runtime-promotion-monitoring.json',
  ]),
  /explicit --apply/
);
assert.equal(
  parseArgs(['--capture', 'capture.json', '--apply']).outputPolicyPath,
  path.join(SOURCE_ROOT, 'tools/policies/runtime-promotion-monitoring.json')
);

console.log('record-runtime-promotion-monitoring.test: ok');
