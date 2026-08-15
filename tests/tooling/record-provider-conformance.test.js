import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  parseArgs,
  recordProviderConformanceCapture,
} from '../../tools/record-provider-conformance.js';

const SOURCE_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const NOW = new Date('2026-08-15T13:00:00.000Z');
const ARTIFACT_ID = `sha256:${'a'.repeat(64)}`;
const EXECUTION_ID = `sha256:${'b'.repeat(64)}`;
const ENVIRONMENT_ID = `sha256:${'c'.repeat(64)}`;

async function writeJson(filePath, value) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

async function writeText(filePath, value = 'fixture\n') {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, value, 'utf8');
}

function providerReceipt(overrides = {}) {
  return {
    receiptVersion: 'doppler_provider_receipt_v1',
    receiptId: 'provider-receipt-fixture',
    source: 'local',
    policyMode: 'local-only',
    policyId: null,
    model: { id: 'qwen-fixture', hash: ARTIFACT_ID, fallbackId: null },
    device: {
      vendor: 'fixture-vendor',
      architecture: 'fixture-architecture',
      device: 'fixture-device',
      description: 'fixture adapter',
      hasF16: true,
      hasSubgroups: false,
      maxBufferSize: 4096,
      submitProbeMs: 0.5,
      deviceEpoch: 1,
    },
    failure: null,
    fallbackDecision: null,
    localDurationMs: 10,
    fallbackDurationMs: null,
    totalDurationMs: 10,
    timestamp: '2026-08-15T12:00:00.000Z',
    diagnoseArtifactRef: null,
    resolutionStatus: 'resolved',
    resolution: {
      schema: 'doppler.resolution-identity/v1',
      logicalModelId: 'qwen-3-5-0-8b-q4k-ehaf16',
      resolvedArtifactVariantId: ARTIFACT_ID,
      resolvedExecutionId: EXECUTION_ID,
    },
    resolutionUnavailableReason: null,
    ...overrides,
  };
}

function capture(overrides = {}) {
  return {
    schema: 'doppler.provider-conformance-capture/v1',
    suiteId: 'qwen35-generation-browser-node',
    laneId: 'browser-webgpu',
    implementationId: 'chromium-webgpu',
    environmentFingerprint: ENVIRONMENT_ID,
    operations: ['prefill', 'decode'],
    lifecycle: { load: 'passed', execute: 'passed', unload: 'passed' },
    correctness: { class: 'exact-token', passed: true },
    qualifiedAtUtc: '2026-08-15T12:05:00.000Z',
    expiresAtUtc: '2026-09-14T12:05:00.000Z',
    evidence: {
      modelContract: 'reports/model-contract.json',
      resolutionIdentity: 'reports/resolution.json',
      operations: 'reports/operations.json',
      lifecycle: 'reports/lifecycle.json',
      correctness: 'reports/correctness.json',
      providerReceipt: 'reports/provider-receipt.json',
    },
    ...overrides,
  };
}

async function createFixture() {
  const repoRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-provider-record-'));
  const policy = JSON.parse(await fs.readFile(
    path.join(SOURCE_ROOT, 'tools/policies/provider-conformance.json'),
    'utf8'
  ));
  const policyPath = path.join(repoRoot, 'tools/policies/provider-conformance.json');
  const capturePath = path.join(repoRoot, 'capture.json');
  const outputPolicyPath = path.join(repoRoot, 'provider-conformance.next.json');
  await writeJson(policyPath, policy);
  for (const lane of policy.providerLanes) {
    await writeText(path.join(repoRoot, lane.contractPath));
  }
  for (const suite of policy.suites) {
    await writeText(path.join(repoRoot, suite.workloadContractPath));
  }
  for (const evidencePath of Object.values(capture().evidence)) {
    await writeJson(path.join(repoRoot, evidencePath), { fixture: true });
  }
  await writeJson(path.join(repoRoot, capture().evidence.providerReceipt), providerReceipt());
  await writeJson(capturePath, capture());
  return { repoRoot, policyPath, capturePath, outputPolicyPath };
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
  const result = await recordProviderConformanceCapture({ ...fixture, now: NOW });
  assert.equal(result.claimAllowed, false);
  assert.equal(result.positiveCapture, true);
  assert.equal(result.resolvedArtifactVariantId, ARTIFACT_ID);
  const output = JSON.parse(await fs.readFile(fixture.outputPolicyPath, 'utf8'));
  const suite = output.suites.find((entry) => entry.id === capture().suiteId);
  assert.equal(suite.resolvedArtifactVariantId, ARTIFACT_ID);
  assert.equal(suite.providers.length, 1);
  assert.equal(suite.providers[0].resolvedExecutionId, EXECUTION_ID);
  assert.equal(suite.providers[0].claimAllowed, false);
  assert.deepEqual(suite.providers[0].blockers, [
    'provider-capture-awaiting-explicit-promotion',
  ]);
  assert.ok(!suite.blockers.includes('browser-provider-qualification-receipt-missing'));
  assert.ok(suite.blockers.includes('browser-webgpu-provider-candidate-recorded-not-promoted'));
});

await withFixture(async (fixture) => {
  const receiptPath = path.join(fixture.repoRoot, capture().evidence.providerReceipt);
  const mismatched = providerReceipt({
    resolution: {
      ...providerReceipt().resolution,
      logicalModelId: 'different-logical-model',
    },
  });
  await writeJson(receiptPath, mismatched);
  await assert.rejects(
    () => recordProviderConformanceCapture({ ...fixture, now: NOW }),
    /does not match suite/
  );
  await assert.rejects(() => fs.stat(fixture.outputPolicyPath));
});

await withFixture(async (fixture) => {
  const receiptPath = path.join(fixture.repoRoot, capture().evidence.providerReceipt);
  await writeJson(receiptPath, providerReceipt({
    source: 'fallback',
    fallbackDecision: { reason: 'fixture', eligible: true, executed: true, deniedReason: null },
  }));
  await assert.rejects(
    () => recordProviderConformanceCapture({ ...fixture, now: NOW }),
    /cannot record a fallback provider receipt/
  );
});

await withFixture(async (fixture) => {
  const receiptPath = path.join(fixture.repoRoot, capture().evidence.providerReceipt);
  await writeJson(receiptPath, providerReceipt({
    failure: {
      failureClass: 'gpu_device_lost',
      failureCode: 'DOPPLER_GPU_DEVICE_LOST',
      stage: 'load',
      surface: 'webgpu',
      device: null,
      modelId: null,
      runtimeProfile: null,
      kernelPathId: null,
      isSimulated: false,
      message: 'fixture failure',
    },
    resolutionStatus: 'unavailable',
    resolution: null,
    resolutionUnavailableReason: 'load-failed-before-resolution',
  }));
  await writeJson(fixture.capturePath, capture({
    lifecycle: { load: 'failed', execute: 'not-run', unload: 'passed' },
    correctness: { class: 'exact-token', passed: false },
  }));
  const result = await recordProviderConformanceCapture({ ...fixture, now: NOW });
  assert.equal(result.positiveCapture, false);
  assert.equal(result.resolvedArtifactVariantId, null);
  assert.ok(result.blockers.includes('provider-receipt-recorded-failure'));
  assert.ok(result.blockers.includes('resolution-identity-unavailable'));
});

await withFixture(async (fixture) => {
  await recordProviderConformanceCapture({ ...fixture, now: NOW });
  await fs.copyFile(fixture.outputPolicyPath, fixture.policyPath);
  await assert.rejects(
    () => recordProviderConformanceCapture({ ...fixture, now: NOW }),
    /already contains provider lane browser-webgpu/
  );
  const replaced = await recordProviderConformanceCapture({
    ...fixture,
    now: NOW,
    replace: true,
  });
  assert.equal(replaced.claimAllowed, false);
});

assert.throws(
  () => parseArgs(['--capture', 'capture.json', '--out', 'tools/policies/provider-conformance.json']),
  /explicit --apply/
);
assert.equal(
  parseArgs(['--capture', 'capture.json', '--apply']).outputPolicyPath,
  path.join(SOURCE_ROOT, 'tools/policies/provider-conformance.json')
);

console.log('record-provider-conformance.test: ok');
