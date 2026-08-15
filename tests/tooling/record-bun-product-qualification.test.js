import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  parseArgs,
  recordBunProductQualification,
} from '../../tools/record-bun-product-qualification.js';
import {
  BUN_QUALIFICATION_EVIDENCE_CLASSES,
} from '../../tools/lib/bun-product-qualification-evidence.js';

const SOURCE_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const NOW = new Date('2026-08-16T00:00:00.000Z');
const ARTIFACT_ID = `sha256:${'a'.repeat(64)}`;
const EXECUTION_ID = `sha256:${'b'.repeat(64)}`;
const ENVIRONMENT_ID = `sha256:${'c'.repeat(64)}`;
const HELD_OUT_ID = `sha256:${'d'.repeat(64)}`;
const DISCOVERY_ID = `sha256:${'e'.repeat(64)}`;
const COMPARISON_ID = `sha256:${'f'.repeat(64)}`;
const HARNESS_REVISION = '1'.repeat(40);
const RUNTIME_REVISION = '2'.repeat(40);

async function writeJson(filePath, value) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function capture(overrides = {}) {
  return {
    schema: 'doppler.bun-product-qualification-capture/v1',
    qualificationId: 'qwen35-generation-bun-product',
    qualifiedAtUtc: '2026-08-15T21:00:00.000Z',
    expiresAtUtc: '2026-09-14T21:00:00.000Z',
    evidence: {
      execution: 'evidence/execution.json',
      surfaceConformance: 'evidence/surface-conformance.json',
      lifecycle: 'evidence/lifecycle.json',
      correctnessQuality: 'evidence/correctness-quality.json',
      reliability: 'evidence/reliability.json',
      memory: 'evidence/memory.json',
      coldWarmResponse: 'evidence/cold-warm.json',
      incumbentControl: 'evidence/incumbent.json',
      upgradeRequalification: 'evidence/upgrade.json',
      rollbackRevocation: 'evidence/rollback.json',
    },
    ...overrides,
  };
}

function executionReceipt(logicalModelId) {
  return {
    receiptVersion: 'doppler_provider_receipt_v1',
    receiptId: 'bun-generation-execution',
    source: 'local',
    model: { id: logicalModelId, hash: ARTIFACT_ID },
    device: { vendor: 'fixture-vendor' },
    failure: null,
    fallbackDecision: null,
    timestamp: '2026-08-15T19:00:00.000Z',
    resolutionStatus: 'resolved',
    resolution: {
      schema: 'doppler.resolution-identity/v1',
      logicalModelId,
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

function semanticReceipt(qualification, evidenceClass, overrides = {}) {
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
    capturedAtUtc: '2026-08-15T20:00:00.000Z',
    result: { passed: true },
    observations: observationsFor(evidenceClass, qualification.correctnessClass),
    ...overrides,
  };
}

async function createFixture() {
  const repoRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-bun-record-'));
  const policy = JSON.parse(await fs.readFile(
    path.join(SOURCE_ROOT, 'tools/policies/bun-product-qualification.json'),
    'utf8'
  ));
  const policyPath = path.join(repoRoot, 'tools/policies/bun-product-qualification.json');
  const capturePath = path.join(repoRoot, 'capture.json');
  const outputPolicyPath = path.join(repoRoot, 'bun-product.next.json');
  const qualification = policy.qualifications[0];
  const captureValue = capture();
  await writeJson(policyPath, policy);
  await writeJson(
    path.join(repoRoot, 'src/config/support-tiers/subsystems.json'),
    { subsystems: [{ id: 'runtime.bun-webgpu', tier: 'experimental' }] }
  );
  await writeJson(
    path.join(repoRoot, 'benchmarks/vendors/registry.json'),
    { products: [{ id: 'doppler-bun', status: 'experimental' }] }
  );
  await writeJson(
    path.join(repoRoot, 'benchmarks/vendors/release-matrix.json'),
    { targets: [{ id: 'doppler-bun', status: 'experimental' }] }
  );
  await writeJson(
    path.join(repoRoot, captureValue.evidence.execution),
    executionReceipt(qualification.logicalModelId)
  );
  for (const evidenceClass of BUN_QUALIFICATION_EVIDENCE_CLASSES) {
    await writeJson(
      path.join(repoRoot, captureValue.evidence[evidenceClass]),
      semanticReceipt(qualification, evidenceClass)
    );
  }
  await writeJson(capturePath, captureValue);
  return { repoRoot, policyPath, capturePath, outputPolicyPath, qualification };
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
  const result = await recordBunProductQualification({ ...fixture, now: NOW });
  assert.equal(result.claimAllowed, false);
  assert.equal(result.resolvedArtifactVariantId, ARTIFACT_ID);
  assert.equal(result.resolvedExecutionId, EXECUTION_ID);
  assert.equal(result.bunVersion, '1.2.1');
  assert.equal(result.webgpuImplementationId, 'bun-native-webgpu');
  assert.equal(result.providerId, 'bun:webgpu');
  assert.ok(result.blockers.includes('bun-product-evaluation-awaiting-explicit-promotion'));
  assert.ok(result.blockers.includes('bun-promotion-evidence-missing'));
  const output = JSON.parse(await fs.readFile(fixture.outputPolicyPath, 'utf8'));
  const qualification = output.qualifications[0];
  assert.equal(qualification.claimAllowed, false);
  assert.equal(qualification.evidence.promotion, null);
  assert.match(qualification.evidence.execution.digest, /^sha256:[0-9a-f]{64}$/);
  assert.match(qualification.evidence.memory.digest, /^sha256:[0-9a-f]{64}$/);
});

await withFixture(async (fixture) => {
  const receiptPath = path.join(fixture.repoRoot, capture().evidence.lifecycle);
  await writeJson(receiptPath, semanticReceipt(fixture.qualification, 'lifecycle', {
    bunVersion: 'different-bun-version',
  }));
  await assert.rejects(
    () => recordBunProductQualification({ ...fixture, now: NOW }),
    /lifecycle evidence is invalid.*bunVersion does not match/
  );
  await assert.rejects(() => fs.stat(fixture.outputPolicyPath));
});

await withFixture(async (fixture) => {
  await writeJson(
    path.join(fixture.repoRoot, capture().evidence.memory),
    { fixture: true }
  );
  await assert.rejects(
    () => recordBunProductQualification({ ...fixture, now: NOW }),
    /memory evidence is invalid.*schema is required/
  );
});

await withFixture(async (fixture) => {
  await recordBunProductQualification({ ...fixture, now: NOW });
  await fs.copyFile(fixture.outputPolicyPath, fixture.policyPath);
  await assert.rejects(
    () => recordBunProductQualification({ ...fixture, now: NOW }),
    /already has evaluation state/
  );
  const result = await recordBunProductQualification({
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
    'tools/policies/bun-product-qualification.json',
  ]),
  /explicit --apply/
);
assert.equal(
  parseArgs(['--capture', 'capture.json', '--apply']).outputPolicyPath,
  path.join(SOURCE_ROOT, 'tools/policies/bun-product-qualification.json')
);

console.log('record-bun-product-qualification.test: ok');
