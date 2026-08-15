import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import {
  buildProductIntegrationQualificationReport,
  validateProductIntegrationQualification,
} from '../../tools/check-product-integration-qualification.js';
import { computeCanonicalJsonSha256 } from '../../tools/lib/canonical-json.js';

const POLICY_PATH = path.join(
  process.cwd(),
  'tools',
  'policies',
  'product-integration-qualification.json'
);
const NOW = new Date('2026-08-15T12:00:00.000Z');
const ARTIFACT_ID = `sha256:${'a'.repeat(64)}`;
const EXECUTION_ID = `sha256:${'b'.repeat(64)}`;
const ENVIRONMENT_ID = `sha256:${'c'.repeat(64)}`;
const COMPARISON_ID = `sha256:${'d'.repeat(64)}`;
const APPLICATION_REVISION = '1'.repeat(40);
const HARNESS_REVISION = '2'.repeat(40);
const TEST_ROOT = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-product-integration-'));

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

async function writeJson(relativePath, value) {
  const filePath = path.join(TEST_ROOT, relativePath);
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
  return {
    path: relativePath,
    digest: computeCanonicalJsonSha256(value),
  };
}

function baseIntegration(id, applicationName, workload) {
  return {
    id,
    applicationName,
    workload,
    owner: `${id}-owner`,
    ownerConfirmedAtUtc: '2026-08-01T00:00:00.000Z',
    qualificationLevel: 'product-supported',
    lifecycle: 'active',
    claimAllowed: true,
    logicalModelId: `${id}-logical-model`,
    resolvedArtifactVariantId: ARTIFACT_ID,
    resolvedExecutionId: EXECUTION_ID,
    qualifiedAtUtc: '2026-08-01T01:00:00.000Z',
    expiresAtUtc: '2026-10-01T00:00:00.000Z',
    evidence: {},
    blockers: [],
  };
}

function ownerConfirmation(integration) {
  return {
    schema: 'doppler.product-integration-owner-confirmation/v1',
    integrationId: integration.id,
    applicationName: integration.applicationName,
    workload: integration.workload,
    owner: integration.owner,
    ownerRepository: `clocksmith/${integration.id}`,
    applicationRevision: APPLICATION_REVISION,
    confirmedAtUtc: integration.ownerConfirmedAtUtc,
    maintenanceStatus: 'active',
    statement: 'The named owner confirms active maintenance of this integration.',
  };
}

function identityReceipt(integration) {
  return {
    receiptVersion: 'doppler_provider_receipt_v1',
    receiptId: `${integration.id}-identity-receipt`,
    source: 'local',
    model: { id: integration.logicalModelId, hash: ARTIFACT_ID },
    device: { vendor: 'fixture-vendor' },
    failure: null,
    fallbackDecision: null,
    timestamp: '2026-08-01T00:30:00.000Z',
    resolutionStatus: 'resolved',
    resolution: {
      schema: 'doppler.resolution-identity/v1',
      logicalModelId: integration.logicalModelId,
      resolvedArtifactVariantId: ARTIFACT_ID,
      resolvedExecutionId: EXECUTION_ID,
    },
  };
}

function observationsFor(evidenceClass) {
  const observations = {
    installToFirstVerifiedOutput: {
      surface: 'root-api',
      installSucceeded: true,
      firstVerifiedOutputMs: 1000,
      maximumFirstVerifiedOutputMs: 2000,
    },
    sourceTaskQualityRetention: {
      sourceScore: 100,
      dopplerScore: 95,
      retentionRatio: 0.95,
      minimumRetentionRatio: 0.9,
    },
    reliability: {
      attempts: 100,
      successes: 99,
      minimumSuccessRate: 0.99,
      crashes: 0,
      maximumCrashes: 0,
      ooms: 0,
      maximumOoms: 0,
      deviceLosses: 0,
      maximumDeviceLosses: 0,
    },
    memory: {
      peakBytes: 1000,
      budgetBytes: 2000,
    },
    coldWarmResponse: {
      sampleCount: 20,
      coldP50Ms: 100,
      coldP95Ms: 150,
      coldP95LimitMs: 200,
      warmP50Ms: 20,
      warmP95Ms: 30,
      warmP95LimitMs: 50,
    },
    browserHardwareQualification: {
      qualifiedTargets: ['chromium-rdna3', 'chromium-apple-m3'],
      failedTargets: [],
      minimumQualifiedTargets: 2,
    },
    incumbentControl: {
      incumbentProviderId: 'instrumented-incumbent',
      incumbentArtifactId: 'incumbent-artifact@revision',
      comparisonReceiptDigest: COMPARISON_ID,
      incumbentAvailable: true,
      correctnessComparable: true,
    },
    upgradeRequalification: {
      fromVersion: '0.5.0',
      toVersion: '0.5.1',
      migrationSucceeded: true,
      identityPreserved: true,
      taskGatePassed: true,
    },
    rollbackRevocation: {
      knownSafeVersion: '0.5.0',
      rollbackSucceeded: true,
      revocationObserved: true,
      taskGatePassed: true,
    },
  };
  return observations[evidenceClass];
}

function outcomeEvidence(integration, evidenceClass, overrides = {}) {
  return {
    schema: 'doppler.product-integration-evidence/v1',
    evidenceClass,
    integrationId: integration.id,
    applicationName: integration.applicationName,
    workload: integration.workload,
    owner: integration.owner,
    applicationRevision: APPLICATION_REVISION,
    harnessRevision: HARNESS_REVISION,
    environmentFingerprint: ENVIRONMENT_ID,
    logicalModelId: integration.logicalModelId,
    resolvedArtifactVariantId: ARTIFACT_ID,
    resolvedExecutionId: EXECUTION_ID,
    capturedAtUtc: '2026-08-01T00:45:00.000Z',
    result: { passed: true },
    observations: observationsFor(evidenceClass),
    ...overrides,
  };
}

async function preparedIntegration(id, applicationName, workload) {
  const integration = baseIntegration(id, applicationName, workload);
  integration.evidence.ownerConfirmation = await writeJson(
    `evidence/${id}-owner-confirmation.json`,
    ownerConfirmation(integration)
  );
  integration.evidence.identity = await writeJson(
    `evidence/${id}-identity.json`,
    identityReceipt(integration)
  );
  for (const evidenceClass of [
    'installToFirstVerifiedOutput',
    'sourceTaskQualityRetention',
    'reliability',
    'memory',
    'coldWarmResponse',
    'browserHardwareQualification',
    'incumbentControl',
    'upgradeRequalification',
    'rollbackRevocation',
  ]) {
    integration.evidence[evidenceClass] = await writeJson(
      `evidence/${id}-${evidenceClass}.json`,
      outcomeEvidence(integration, evidenceClass)
    );
  }
  return integration;
}

const policy = JSON.parse(await fs.readFile(POLICY_PATH, 'utf8'));

{
  const report = await buildProductIntegrationQualificationReport({
    policyPath: POLICY_PATH,
    now: NOW,
  });
  assert.equal(report.ok, true, report.errors.join('\n'));
  assert.equal(report.gateSatisfied, false);
  assert.equal(report.qualifiedIntegrations, 0);
  assert.equal(report.candidateIntegrations, 3);
  assert.deepEqual(report.candidateWorkloads, [
    'generation',
    'embedding-retrieval',
    'reranking',
  ]);
  assert.ok(report.integrations.every((entry) => entry.qualified === false));
}

{
  const complete = clone(policy);
  complete.integrations = await Promise.all([
    preparedIntegration('private-chat', 'Private Chat', 'generation'),
    preparedIntegration('local-search', 'Local Search', 'embedding-retrieval'),
    preparedIntegration('result-ranking', 'Result Ranking', 'reranking'),
  ]);
  const report = await validateProductIntegrationQualification(complete, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.deepEqual(report.errors, []);
  assert.equal(report.qualifiedIntegrations, 3);
  assert.equal(report.distinctQualifiedApplications, 3);
  assert.equal(report.gateSatisfied, true);
}

{
  const stale = clone(policy);
  const record = await preparedIntegration('stale-chat', 'Stale Chat', 'generation');
  record.ownerConfirmedAtUtc = '2025-01-01T00:00:00.000Z';
  stale.integrations = [record];
  const report = await validateProductIntegrationQualification(stale, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(
    report.errors.includes(
      'stale-chat: claimAllowed integration does not satisfy product qualification'
    ),
    report.errors.join('\n')
  );
  assert.equal(report.gateSatisfied, false);
}

{
  const candidate = clone(policy);
  const record = await preparedIntegration('candidate-chat', 'Candidate Chat', 'generation');
  record.qualificationLevel = 'runtime-verified';
  record.lifecycle = 'candidate';
  record.claimAllowed = false;
  record.blockers = ['held-out-task-gate-missing'];
  record.evidence.sourceTaskQualityRetention = null;
  candidate.integrations = [record];
  const report = await validateProductIntegrationQualification(candidate, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.deepEqual(report.errors, []);
  assert.equal(report.integrations[0].qualified, false);
  assert.equal(report.candidateIntegrations, 1);
  assert.ok(report.integrations[0].missingEvidence.includes('sourceTaskQualityRetention'));
}

{
  const conflated = clone(policy);
  conflated.integrations[0].resolvedArtifactVariantId = 'named-manifest-variant';
  const report = await validateProductIntegrationQualification(conflated, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(
    report.errors.includes(
      'reploid-local-generation: resolvedArtifactVariantId must be a SHA-256 identity or null'
    ),
    report.errors.join('\n')
  );
}

{
  const tampered = clone(policy);
  const record = await preparedIntegration('tampered-chat', 'Tampered Chat', 'generation');
  const evidenceRef = record.evidence.memory;
  const receipt = JSON.parse(await fs.readFile(path.join(TEST_ROOT, evidenceRef.path), 'utf8'));
  receipt.observations.peakBytes = 1500;
  await fs.writeFile(
    path.join(TEST_ROOT, evidenceRef.path),
    `${JSON.stringify(receipt, null, 2)}\n`,
    'utf8'
  );
  tampered.integrations = [record];
  const report = await validateProductIntegrationQualification(tampered, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(
    report.errors.some((error) => error.includes('digest does not match canonical JSON evidence')),
    report.errors.join('\n')
  );
}

{
  const falsePass = clone(policy);
  const record = await preparedIntegration('false-pass-chat', 'False Pass Chat', 'generation');
  const evidenceRef = record.evidence.reliability;
  const receipt = JSON.parse(await fs.readFile(path.join(TEST_ROOT, evidenceRef.path), 'utf8'));
  receipt.observations.successes = 50;
  record.evidence.reliability = await writeJson(evidenceRef.path, receipt);
  falsePass.integrations = [record];
  const report = await validateProductIntegrationQualification(falsePass, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(
    report.errors.some((error) => error.includes('result.passed does not match')),
    report.errors.join('\n')
  );
}

{
  const wrongIdentity = clone(policy);
  const record = await preparedIntegration('wrong-id-chat', 'Wrong ID Chat', 'generation');
  const evidenceRef = record.evidence.identity;
  const receipt = JSON.parse(await fs.readFile(path.join(TEST_ROOT, evidenceRef.path), 'utf8'));
  receipt.resolution.resolvedExecutionId = `sha256:${'f'.repeat(64)}`;
  record.evidence.identity = await writeJson(evidenceRef.path, receipt);
  wrongIdentity.integrations = [record];
  const report = await validateProductIntegrationQualification(wrongIdentity, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(
    report.errors.some((error) => error.includes('Doppler execution identity')),
    report.errors.join('\n')
  );
}

{
  const prePromotionMismatch = clone(policy);
  const record = await preparedIntegration(
    'pre-promotion-chat',
    'Pre-promotion Chat',
    'generation'
  );
  record.resolvedArtifactVariantId = null;
  record.resolvedExecutionId = null;
  record.qualificationLevel = 'contract-ready';
  record.lifecycle = 'candidate';
  record.claimAllowed = false;
  record.blockers = ['application-evaluation-awaiting-explicit-promotion'];
  const evidenceRef = record.evidence.reliability;
  const receipt = JSON.parse(await fs.readFile(path.join(TEST_ROOT, evidenceRef.path), 'utf8'));
  receipt.resolvedExecutionId = `sha256:${'e'.repeat(64)}`;
  record.evidence.reliability = await writeJson(evidenceRef.path, receipt);
  prePromotionMismatch.integrations = [record];
  const report = await validateProductIntegrationQualification(prePromotionMismatch, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(
    report.errors.some((error) => error.includes('does not match expected')),
    report.errors.join('\n')
  );
}

await fs.rm(TEST_ROOT, { recursive: true, force: true });

console.log('product-integration-qualification.test: ok');
