import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { computeCanonicalJsonSha256 } from '../../tools/lib/canonical-json.js';
import { PRODUCT_OUTCOME_EVIDENCE_CLASSES } from '../../tools/lib/product-integration-evidence.js';
import {
  parseArgs,
  recordProductIntegrationEvaluation,
} from '../../tools/record-product-integration-evaluation.js';

const SOURCE_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const NOW = new Date('2026-08-16T00:00:00.000Z');
const ARTIFACT_ID = `sha256:${'a'.repeat(64)}`;
const EXECUTION_ID = `sha256:${'b'.repeat(64)}`;
const ENVIRONMENT_ID = `sha256:${'c'.repeat(64)}`;
const COMPARISON_ID = `sha256:${'d'.repeat(64)}`;
const APPLICATION_REVISION = '1'.repeat(40);
const HARNESS_REVISION = '2'.repeat(40);
const INTEGRATION_ID = 'reploid-local-generation';

async function writeJson(filePath, value) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function ownerConfirmation(integration, overrides = {}) {
  return {
    schema: 'doppler.product-integration-owner-confirmation/v1',
    integrationId: integration.id,
    applicationName: integration.applicationName,
    workload: integration.workload,
    owner: integration.owner,
    ownerRepository: integration.owner,
    applicationRevision: APPLICATION_REVISION,
    confirmedAtUtc: '2026-08-15T18:00:00.000Z',
    maintenanceStatus: 'active',
    statement: 'The named owner confirms active maintenance of this integration.',
    ...overrides,
  };
}

function identityReceipt(integration, overrides = {}) {
  return {
    receiptVersion: 'doppler_provider_receipt_v1',
    receiptId: `${integration.id}-identity`,
    source: 'local',
    model: { id: integration.logicalModelId, hash: ARTIFACT_ID },
    device: { vendor: 'fixture-vendor' },
    failure: null,
    fallbackDecision: null,
    timestamp: '2026-08-15T18:30:00.000Z',
    resolutionStatus: 'resolved',
    resolution: {
      schema: 'doppler.resolution-identity/v1',
      logicalModelId: integration.logicalModelId,
      resolvedArtifactVariantId: ARTIFACT_ID,
      resolvedExecutionId: EXECUTION_ID,
    },
    ...overrides,
  };
}

function observationsFor(evidenceClass) {
  return {
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
    memory: { peakBytes: 1000, budgetBytes: 2000 },
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
  }[evidenceClass];
}

function outcomeReceipt(integration, evidenceClass, overrides = {}) {
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
    capturedAtUtc: '2026-08-15T19:00:00.000Z',
    result: { passed: true },
    observations: observationsFor(evidenceClass),
    ...overrides,
  };
}

function capture(evidencePaths) {
  return {
    schema: 'doppler.product-integration-evaluation-capture/v1',
    integrationId: INTEGRATION_ID,
    evaluatedAtUtc: '2026-08-15T20:00:00.000Z',
    expiresAtUtc: '2026-09-14T20:00:00.000Z',
    evidencePaths,
  };
}

async function createFixture() {
  const repoRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-product-record-'));
  const policy = JSON.parse(await fs.readFile(
    path.join(SOURCE_ROOT, 'tools/policies/product-integration-qualification.json'),
    'utf8'
  ));
  const integration = policy.integrations.find((entry) => entry.id === INTEGRATION_ID);
  const policyPath = path.join(repoRoot, 'tools/policies/product-integration-qualification.json');
  const capturePath = path.join(repoRoot, 'capture.json');
  const outputPolicyPath = path.join(repoRoot, 'product-integration.next.json');
  const evidencePaths = {
    ownerConfirmation: 'evidence/owner-confirmation.json',
    installToFirstVerifiedOutput: 'evidence/install.json',
    identity: 'evidence/identity.json',
    sourceTaskQualityRetention: 'evidence/quality.json',
    reliability: 'evidence/reliability.json',
    memory: 'evidence/memory.json',
    coldWarmResponse: 'evidence/cold-warm.json',
    browserHardwareQualification: 'evidence/browser-hardware.json',
    incumbentControl: 'evidence/incumbent.json',
    upgradeRequalification: 'evidence/upgrade.json',
    rollbackRevocation: 'evidence/rollback.json',
  };
  await writeJson(policyPath, policy);
  await writeJson(path.join(repoRoot, evidencePaths.ownerConfirmation), ownerConfirmation(integration));
  await writeJson(path.join(repoRoot, evidencePaths.identity), identityReceipt(integration));
  for (const evidenceClass of PRODUCT_OUTCOME_EVIDENCE_CLASSES) {
    await writeJson(
      path.join(repoRoot, evidencePaths[evidenceClass]),
      outcomeReceipt(integration, evidenceClass)
    );
  }
  await writeJson(capturePath, capture(evidencePaths));
  return {
    repoRoot,
    policyPath,
    capturePath,
    outputPolicyPath,
    evidencePaths,
    integration,
  };
}

async function withFixture(run) {
  const fixture = await createFixture();
  try {
    await run(fixture);
  } finally {
    await fs.rm(fixture.repoRoot, { recursive: true, force: true });
  }
}

await withFixture(async (fixture) => {
  const result = await recordProductIntegrationEvaluation({ ...fixture, now: NOW });
  assert.equal(result.claimAllowed, false);
  assert.equal(result.qualificationLevel, 'contract-ready');
  assert.equal(result.lifecycle, 'candidate');
  assert.equal(result.ownerConfirmedAtUtc, '2026-08-15T18:00:00.000Z');
  assert.equal(result.resolvedArtifactVariantId, ARTIFACT_ID);
  assert.equal(result.resolvedExecutionId, EXECUTION_ID);
  assert.deepEqual(result.blockers, ['application-evaluation-awaiting-explicit-promotion']);
  const output = JSON.parse(await fs.readFile(fixture.outputPolicyPath, 'utf8'));
  const integration = output.integrations.find((entry) => entry.id === INTEGRATION_ID);
  assert.equal(integration.claimAllowed, false);
  assert.equal(integration.qualificationLevel, 'contract-ready');
  for (const [field, evidencePath] of Object.entries(fixture.evidencePaths)) {
    const receipt = JSON.parse(await fs.readFile(path.join(fixture.repoRoot, evidencePath), 'utf8'));
    assert.deepEqual(integration.evidence[field], {
      path: evidencePath,
      digest: computeCanonicalJsonSha256(receipt),
    });
  }
});

await withFixture(async (fixture) => {
  const reliabilityPath = path.join(fixture.repoRoot, fixture.evidencePaths.reliability);
  await writeJson(reliabilityPath, outcomeReceipt(fixture.integration, 'reliability', {
    observations: { ...observationsFor('reliability'), successes: 98 },
  }));
  await assert.rejects(
    () => recordProductIntegrationEvaluation({ ...fixture, now: NOW }),
    /reliability evidence is invalid.*result.passed does not match its observations/
  );
});

await withFixture(async (fixture) => {
  const identityPath = path.join(fixture.repoRoot, fixture.evidencePaths.identity);
  await writeJson(identityPath, identityReceipt(fixture.integration, {
    resolutionStatus: 'unavailable',
    resolution: null,
  }));
  const reliabilityPath = path.join(fixture.repoRoot, fixture.evidencePaths.reliability);
  await writeJson(reliabilityPath, outcomeReceipt(fixture.integration, 'reliability', {
    resolvedExecutionId: `sha256:${'e'.repeat(64)}`,
  }));
  await assert.rejects(
    () => recordProductIntegrationEvaluation({ ...fixture, now: NOW }),
    /reliability evidence is invalid.*does not match expected/
  );
});

await withFixture(async (fixture) => {
  const reliabilityPath = path.join(fixture.repoRoot, fixture.evidencePaths.reliability);
  await writeJson(reliabilityPath, outcomeReceipt(fixture.integration, 'reliability', {
    result: { passed: false },
    observations: { ...observationsFor('reliability'), successes: 98 },
  }));
  const result = await recordProductIntegrationEvaluation({ ...fixture, now: NOW });
  assert.ok(result.blockers.includes('reliability-not-passed'));
});

await withFixture(async (fixture) => {
  const ownerPath = path.join(fixture.repoRoot, fixture.evidencePaths.ownerConfirmation);
  await writeJson(ownerPath, ownerConfirmation(fixture.integration, { owner: 'clocksmith/other' }));
  await assert.rejects(
    () => recordProductIntegrationEvaluation({ ...fixture, now: NOW }),
    /Owner confirmation is invalid.*does not match expected/
  );
});

await withFixture(async (fixture) => {
  const reliabilityPath = path.join(fixture.repoRoot, fixture.evidencePaths.reliability);
  await writeJson(reliabilityPath, outcomeReceipt(fixture.integration, 'reliability', {
    resolvedExecutionId: `sha256:${'e'.repeat(64)}`,
  }));
  await assert.rejects(
    () => recordProductIntegrationEvaluation({ ...fixture, now: NOW }),
    /reliability evidence is invalid.*does not match expected/
  );
});

await withFixture(async (fixture) => {
  await recordProductIntegrationEvaluation({ ...fixture, now: NOW });
  await fs.copyFile(fixture.outputPolicyPath, fixture.policyPath);
  await assert.rejects(
    () => recordProductIntegrationEvaluation({ ...fixture, now: NOW }),
    /already contains evaluation state/
  );
  const result = await recordProductIntegrationEvaluation({
    ...fixture,
    now: NOW,
    replace: true,
  });
  assert.equal(result.claimAllowed, false);
});

assert.throws(
  () => parseArgs([
    '--capture',
    'capture.json',
    '--out',
    'tools/policies/product-integration-qualification.json',
  ]),
  /explicit --apply/
);
assert.equal(
  parseArgs(['--capture', 'capture.json', '--apply']).outputPolicyPath,
  path.join(SOURCE_ROOT, 'tools/policies/product-integration-qualification.json')
);

console.log('record-product-integration-evaluation.test: ok');
