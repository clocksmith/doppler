import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import {
  buildBunProductQualificationReport,
  validateBunProductQualificationPolicy,
} from '../../tools/check-bun-product-qualification.js';
import { computeCanonicalJsonSha256 } from '../../tools/lib/canonical-json.js';
import {
  BUN_QUALIFICATION_EVIDENCE_CLASSES,
  computeBunQualificationEvidenceSetDigest,
} from '../../tools/lib/bun-product-qualification-evidence.js';

const POLICY_PATH = path.join(process.cwd(), 'tools/policies/bun-product-qualification.json');
const NOW = new Date('2026-08-15T12:00:00.000Z');
const ARTIFACT_ID = `sha256:${'a'.repeat(64)}`;
const EXECUTION_ID = `sha256:${'b'.repeat(64)}`;
const ENVIRONMENT_ID = `sha256:${'c'.repeat(64)}`;
const HELD_OUT_ID = `sha256:${'d'.repeat(64)}`;
const DISCOVERY_ID = `sha256:${'e'.repeat(64)}`;
const COMPARISON_ID = `sha256:${'f'.repeat(64)}`;
const HARNESS_REVISION = '1'.repeat(40);
const RUNTIME_REVISION = '2'.repeat(40);
const TEST_ROOT = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-bun-product-'));

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

async function writeJson(relativePath, value) {
  const filePath = path.join(TEST_ROOT, relativePath);
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
  return { path: relativePath, digest: computeCanonicalJsonSha256(value) };
}

function executionReceipt(qualification) {
  return {
    receiptVersion: 'doppler_provider_receipt_v1',
    receiptId: `${qualification.id}-execution`,
    source: 'local',
    model: { id: qualification.logicalModelId, hash: ARTIFACT_ID },
    device: { vendor: 'fixture-vendor' },
    failure: null,
    fallbackDecision: null,
    timestamp: '2026-08-01T00:00:00.000Z',
    resolutionStatus: 'resolved',
    resolution: {
      schema: 'doppler.resolution-identity/v1',
      logicalModelId: qualification.logicalModelId,
      resolvedArtifactVariantId: ARTIFACT_ID,
      resolvedExecutionId: EXECUTION_ID,
    },
  };
}

function observationsFor(evidenceClass, correctnessClass) {
  return {
    surfaceConformance: {
      rootApiPassed: true,
      cliPassed: true,
      semanticParityPassed: true,
      unsupportedFallbackUsed: false,
    },
    lifecycle: {
      loadPassed: true,
      executePassed: true,
      unloadPassed: true,
      repeatedSessions: 10,
      minimumRepeatedSessions: 5,
    },
    correctnessQuality: {
      correctnessClass,
      correctnessPassed: true,
      heldOutSetDigest: HELD_OUT_ID,
      qualityPassed: true,
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
      minimumSampleCount: 10,
      coldP50Ms: 100,
      coldP95Ms: 150,
      coldP95LimitMs: 200,
      warmP50Ms: 20,
      warmP95Ms: 30,
      warmP95LimitMs: 50,
    },
    incumbentControl: {
      outcome: 'compared',
      incumbentProviderId: 'bun-native-incumbent',
      incumbentArtifactId: 'incumbent-artifact@revision',
      discoveryReceiptDigest: DISCOVERY_ID,
      comparisonReceiptDigest: COMPARISON_ID,
      correctnessComparable: true,
    },
    upgradeRequalification: {
      fromBunVersion: '1.2.0',
      toBunVersion: '1.2.1',
      migrationSucceeded: true,
      identityPreserved: true,
      taskGatePassed: true,
    },
    rollbackRevocation: {
      knownSafeRuntimeRevision: RUNTIME_REVISION,
      rollbackSucceeded: true,
      revocationObserved: true,
      taskGatePassed: true,
    },
  }[evidenceClass];
}

function semanticReceipt(qualification, evidenceClass) {
  return {
    schema: 'doppler.bun-product-qualification-evidence/v1',
    evidenceClass,
    qualificationId: qualification.id,
    workload: qualification.workload,
    logicalModelId: qualification.logicalModelId,
    manifestVariantId: qualification.manifestVariantId,
    resolvedArtifactVariantId: ARTIFACT_ID,
    resolvedExecutionId: EXECUTION_ID,
    bunVersion: '1.2.1',
    webgpuImplementationId: 'bun-native-webgpu',
    providerId: 'bun:webgpu',
    harnessRevision: HARNESS_REVISION,
    environmentFingerprint: ENVIRONMENT_ID,
    capturedAtUtc: '2026-08-01T00:15:00.000Z',
    result: { passed: true },
    observations: observationsFor(evidenceClass, qualification.correctnessClass),
  };
}

async function qualifiedRecord(template) {
  const qualification = clone(template);
  qualification.resolvedArtifactVariantId = ARTIFACT_ID;
  qualification.resolvedExecutionId = EXECUTION_ID;
  qualification.bunVersion = '1.2.1';
  qualification.webgpuImplementationId = 'bun-native-webgpu';
  qualification.providerId = 'bun:webgpu';
  qualification.qualifiedAtUtc = '2026-08-01T01:00:00.000Z';
  qualification.expiresAtUtc = '2026-10-01T00:00:00.000Z';
  qualification.claimAllowed = true;
  qualification.blockers = [];
  qualification.evidence.execution = await writeJson(
    `evidence/${qualification.id}-execution.json`,
    executionReceipt(qualification)
  );
  for (const evidenceClass of BUN_QUALIFICATION_EVIDENCE_CLASSES) {
    qualification.evidence[evidenceClass] = await writeJson(
      `evidence/${qualification.id}-${evidenceClass}.json`,
      semanticReceipt(qualification, evidenceClass)
    );
  }
  const evidenceSet = Object.fromEntries(
    Object.entries(qualification.evidence).filter(([field]) => field !== 'promotion')
  );
  qualification.evidence.promotion = await writeJson(
    `evidence/${qualification.id}-promotion.json`,
    {
      schema: 'doppler.bun-product-promotion-evidence/v1',
      qualificationId: qualification.id,
      workload: qualification.workload,
      logicalModelId: qualification.logicalModelId,
      resolvedArtifactVariantId: ARTIFACT_ID,
      resolvedExecutionId: EXECUTION_ID,
      bunVersion: qualification.bunVersion,
      webgpuImplementationId: qualification.webgpuImplementationId,
      providerId: qualification.providerId,
      evidenceSetDigest: computeBunQualificationEvidenceSetDigest(evidenceSet),
      decision: 'promote',
      authority: 'human',
      promotedAtUtc: '2026-08-01T00:30:00.000Z',
      qualifiedAtUtc: qualification.qualifiedAtUtc,
      expiresAtUtc: qualification.expiresAtUtc,
    }
  );
  return qualification;
}

const policy = JSON.parse(await fs.readFile(POLICY_PATH, 'utf8'));
const experimentalSubsystems = {
  subsystems: [{ id: 'runtime.bun-webgpu', tier: 'experimental' }],
};
const experimentalRelease = {
  targets: [{ id: 'doppler-bun', status: 'experimental' }],
};
const experimentalRegistry = {
  products: [{ id: 'doppler-bun', status: 'experimental' }],
};
const activeSubsystems = {
  subsystems: [{ id: 'runtime.bun-webgpu', tier: 'tier1' }],
};
const activeRelease = {
  targets: [{ id: 'doppler-bun', status: 'active' }],
};
const activeRegistry = {
  products: [{ id: 'doppler-bun', status: 'active' }],
};

{
  const report = await buildBunProductQualificationReport({ policyPath: POLICY_PATH, now: NOW });
  assert.equal(report.ok, true, report.errors.join('\n'));
  assert.equal(report.gateSatisfied, false);
  assert.equal(report.qualifiedWorkloads, 0);
  assert.equal(report.candidateWorkloads, 3);
}

{
  const complete = clone(policy);
  complete.qualifications = await Promise.all(policy.qualifications.map(qualifiedRecord));
  const report = await validateBunProductQualificationPolicy(complete, {
    repoRoot: TEST_ROOT,
    now: NOW,
    subsystems: activeSubsystems,
    releaseRegistry: activeRegistry,
    releaseMatrix: activeRelease,
  });
  assert.deepEqual(report.errors, []);
  assert.equal(report.qualifiedWorkloads, 3);
  assert.equal(report.gateSatisfied, true);
}

{
  const partial = clone(policy);
  partial.qualifications[0] = await qualifiedRecord(policy.qualifications[0]);
  const report = await validateBunProductQualificationPolicy(partial, {
    repoRoot: TEST_ROOT,
    now: NOW,
    subsystems: experimentalSubsystems,
    releaseRegistry: experimentalRegistry,
    releaseMatrix: experimentalRelease,
  });
  assert.ok(
    report.errors.includes(
      'Partial Bun product promotion is forbidden; all required workloads must promote together'
    ),
    report.errors.join('\n')
  );
}

{
  const falseResult = clone(policy);
  const record = await qualifiedRecord(policy.qualifications[0]);
  const memoryPath = record.evidence.memory.path;
  const receipt = JSON.parse(await fs.readFile(path.join(TEST_ROOT, memoryPath), 'utf8'));
  receipt.observations.peakBytes = 3000;
  record.evidence.memory = await writeJson(memoryPath, receipt);
  falseResult.qualifications = [record];
  const report = await validateBunProductQualificationPolicy(falseResult, {
    repoRoot: TEST_ROOT,
    now: NOW,
    subsystems: experimentalSubsystems,
    releaseRegistry: experimentalRegistry,
    releaseMatrix: experimentalRelease,
  });
  assert.ok(
    report.errors.some((error) => error.includes('result.passed does not match observations')),
    report.errors.join('\n')
  );
}

{
  const tampered = clone(policy);
  const record = await qualifiedRecord(policy.qualifications[1]);
  const lifecyclePath = record.evidence.lifecycle.path;
  const receipt = JSON.parse(await fs.readFile(path.join(TEST_ROOT, lifecyclePath), 'utf8'));
  receipt.observations.unloadPassed = false;
  await fs.writeFile(
    path.join(TEST_ROOT, lifecyclePath),
    `${JSON.stringify(receipt, null, 2)}\n`,
    'utf8'
  );
  tampered.qualifications = [record];
  const report = await validateBunProductQualificationPolicy(tampered, {
    repoRoot: TEST_ROOT,
    now: NOW,
    subsystems: experimentalSubsystems,
    releaseRegistry: experimentalRegistry,
    releaseMatrix: experimentalRelease,
  });
  assert.ok(
    report.errors.some((error) => error.includes('digest does not match canonical JSON evidence')),
    report.errors.join('\n')
  );
}

{
  const projectionDrift = clone(policy);
  projectionDrift.qualifications = await Promise.all(policy.qualifications.map(qualifiedRecord));
  const report = await validateBunProductQualificationPolicy(projectionDrift, {
    repoRoot: TEST_ROOT,
    now: NOW,
    subsystems: experimentalSubsystems,
    releaseRegistry: experimentalRegistry,
    releaseMatrix: experimentalRelease,
  });
  assert.ok(report.errors.includes('Bun support subsystem tier must be tier1'));
  assert.ok(report.errors.includes('Bun release engine status must be active'));
  assert.ok(report.errors.includes('Bun release-matrix target status must be active'));
}

await fs.rm(TEST_ROOT, { recursive: true, force: true });

console.log('bun-product-qualification.test: ok');
