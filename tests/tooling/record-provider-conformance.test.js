import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { computeCanonicalJsonSha256 } from '../../tools/lib/canonical-json.js';
import {
  PROVIDER_CONFORMANCE_EVIDENCE_CLASSES,
} from '../../tools/lib/provider-conformance-evidence.js';
import {
  parseArgs,
  recordProviderConformanceCapture,
} from '../../tools/record-provider-conformance.js';

const SOURCE_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const NOW = new Date('2026-08-16T00:00:00.000Z');
const ARTIFACT_ID = `sha256:${'a'.repeat(64)}`;
const EXECUTION_ID = `sha256:${'b'.repeat(64)}`;
const ENVIRONMENT_ID = `sha256:${'c'.repeat(64)}`;
const TOKENIZER_ID = `sha256:${'d'.repeat(64)}`;
const GRAPH_ID = `sha256:${'e'.repeat(64)}`;
const POLICY_ID = `sha256:${'f'.repeat(64)}`;
const OUTPUT_ID = `sha256:${'1'.repeat(64)}`;
const HARNESS_REVISION = '2'.repeat(40);

async function writeJson(filePath, value) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

async function writeText(filePath) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, 'fixture\n', 'utf8');
}

function capture(overrides = {}) {
  return {
    schema: 'doppler.provider-conformance-capture/v2',
    suiteId: 'qwen35-generation-browser-node',
    laneId: 'browser-webgpu',
    qualifiedAtUtc: '2026-08-15T21:00:00.000Z',
    expiresAtUtc: '2026-09-14T21:00:00.000Z',
    evidence: {
      modelContract: 'evidence/model-contract.json',
      resolutionIdentity: 'evidence/resolution-identity.json',
      operations: 'evidence/operations.json',
      lifecycle: 'evidence/lifecycle.json',
      correctness: 'evidence/correctness.json',
      providerReceipt: 'evidence/provider-receipt.json',
    },
    ...overrides,
  };
}

function providerReceipt(overrides = {}) {
  return {
    receiptVersion: 'doppler_provider_receipt_v1',
    receiptId: 'provider-receipt-fixture',
    source: 'local',
    model: { id: 'qwen-3-5-0-8b-q4k-ehaf16', hash: ARTIFACT_ID },
    device: { vendor: 'fixture-vendor' },
    failure: null,
    fallbackDecision: null,
    timestamp: '2026-08-15T19:00:00.000Z',
    resolutionStatus: 'resolved',
    resolution: {
      schema: 'doppler.resolution-identity/v1',
      logicalModelId: 'qwen-3-5-0-8b-q4k-ehaf16',
      resolvedArtifactVariantId: ARTIFACT_ID,
      resolvedExecutionId: EXECUTION_ID,
    },
    ...overrides,
  };
}

function observationsFor(evidenceClass) {
  return {
    modelContract: {
      manifestDigest: ARTIFACT_ID,
      tokenizerDigest: TOKENIZER_ID,
      executionGraphDigest: GRAPH_ID,
      runtimePolicyDigest: POLICY_ID,
      artifactValidated: true,
      tokenizerIdentityMatched: true,
      executionGraphIdentityMatched: true,
      runtimePolicyExplicit: true,
    },
    resolutionIdentity: {
      logicalModelResolved: true,
      manifestVariantMatched: true,
      artifactDigestMatched: true,
      executionDigestMatched: true,
      fallbackUsed: false,
    },
    operations: {
      declaredOperations: ['prefill', 'decode'],
      observedOperations: ['prefill', 'decode'],
      unsupportedOperationUsed: false,
    },
    lifecycle: {
      load: 'passed',
      execute: 'passed',
      unload: 'passed',
      repeatedSessions: 5,
      minimumRepeatedSessions: 3,
    },
    correctness: {
      correctnessClass: 'exact-token',
      referenceOutputDigest: OUTPUT_ID,
      providerOutputDigest: OUTPUT_ID,
      tokenParityPassed: true,
      deterministicContinuationPassed: true,
    },
  }[evidenceClass];
}

function semanticReceipt(evidenceClass, receiptDigest, overrides = {}) {
  return {
    schema: 'doppler.provider-conformance-evidence/v1',
    evidenceClass,
    suiteId: 'qwen35-generation-browser-node',
    laneId: 'browser-webgpu',
    workload: 'generation',
    logicalModelId: 'qwen-3-5-0-8b-q4k-ehaf16',
    manifestVariantId: 'qwen-3-5-0-8b-q4k-ehaf16-mv-exec-v1',
    resolvedArtifactVariantId: ARTIFACT_ID,
    resolvedExecutionId: EXECUTION_ID,
    implementationId: 'chromium-webgpu',
    harnessRevision: HARNESS_REVISION,
    environmentFingerprint: ENVIRONMENT_ID,
    providerReceiptDigest: receiptDigest,
    capturedAtUtc: '2026-08-15T20:00:00.000Z',
    result: { passed: true },
    observations: observationsFor(evidenceClass),
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
  for (const lane of policy.providerLanes) await writeText(path.join(repoRoot, lane.contractPath));
  for (const suite of policy.suites) {
    await writeText(path.join(repoRoot, suite.workloadContractPath));
  }
  const captureValue = capture();
  const execution = providerReceipt();
  const receiptDigest = computeCanonicalJsonSha256(execution);
  await writeJson(path.join(repoRoot, captureValue.evidence.providerReceipt), execution);
  for (const evidenceClass of PROVIDER_CONFORMANCE_EVIDENCE_CLASSES) {
    await writeJson(
      path.join(repoRoot, captureValue.evidence[evidenceClass]),
      semanticReceipt(evidenceClass, receiptDigest)
    );
  }
  await writeJson(capturePath, captureValue);
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
  assert.equal(result.resolvedExecutionId, EXECUTION_ID);
  const output = JSON.parse(await fs.readFile(fixture.outputPolicyPath, 'utf8'));
  const suite = output.suites.find((entry) => entry.id === capture().suiteId);
  const provider = suite.providers[0];
  assert.equal(provider.implementationId, 'chromium-webgpu');
  assert.equal(provider.harnessRevision, HARNESS_REVISION);
  assert.equal(provider.environmentFingerprint, ENVIRONMENT_ID);
  assert.deepEqual(provider.operations, ['prefill', 'decode']);
  assert.deepEqual(provider.lifecycle, { load: 'passed', execute: 'passed', unload: 'passed' });
  assert.deepEqual(provider.correctness, { class: 'exact-token', passed: true });
  assert.equal(provider.evidence.promotion, null);
  assert.match(provider.evidence.modelContract.digest, /^sha256:[0-9a-f]{64}$/);
  assert.ok(provider.blockers.includes('provider-promotion-evidence-missing'));
});

await withFixture(async (fixture) => {
  await writeJson(
    path.join(fixture.repoRoot, capture().evidence.modelContract),
    { fixture: true }
  );
  await assert.rejects(
    () => recordProviderConformanceCapture({ ...fixture, now: NOW }),
    /modelContract evidence is invalid.*schema is required/
  );
  await assert.rejects(() => fs.stat(fixture.outputPolicyPath));
});

await withFixture(async (fixture) => {
  const evidencePath = path.join(fixture.repoRoot, capture().evidence.lifecycle);
  const receiptDigest = computeCanonicalJsonSha256(providerReceipt());
  await writeJson(evidencePath, semanticReceipt('lifecycle', receiptDigest, {
    environmentFingerprint: `sha256:${'9'.repeat(64)}`,
  }));
  await assert.rejects(
    () => recordProviderConformanceCapture({ ...fixture, now: NOW }),
    /lifecycle evidence is invalid.*environmentFingerprint does not match/
  );
});

await withFixture(async (fixture) => {
  const evidencePath = path.join(fixture.repoRoot, capture().evidence.lifecycle);
  const receiptDigest = computeCanonicalJsonSha256(providerReceipt());
  const receipt = semanticReceipt('lifecycle', receiptDigest);
  receipt.observations.unload = 'failed';
  receipt.result.passed = false;
  await writeJson(evidencePath, receipt);
  const result = await recordProviderConformanceCapture({ ...fixture, now: NOW });
  assert.equal(result.positiveCapture, false);
  assert.ok(result.blockers.includes('lifecycle-gate-failed'));
  const output = JSON.parse(await fs.readFile(fixture.outputPolicyPath, 'utf8'));
  assert.equal(output.suites[0].providers[0].lifecycle.unload, 'failed');
});

await withFixture(async (fixture) => {
  const receiptPath = path.join(fixture.repoRoot, capture().evidence.providerReceipt);
  const fallbackReceipt = providerReceipt({
    source: 'fallback',
    fallbackDecision: { executed: true },
  });
  await writeJson(receiptPath, fallbackReceipt);
  await assert.rejects(
    () => recordProviderConformanceCapture({ ...fixture, now: NOW }),
    /not a passing local execution/
  );
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
