import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  parseArgs,
  recordRuntimeOwnershipEvaluation,
} from '../../tools/record-runtime-ownership-evaluation.js';

const SOURCE_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const NOW = new Date('2026-08-16T00:00:00.000Z');
const ARTIFACT_ID = `sha256:${'a'.repeat(64)}`;
const EXECUTION_ID = `sha256:${'b'.repeat(64)}`;
const ENVIRONMENT_ID = `sha256:${'c'.repeat(64)}`;
const CONFIGURATION_ID = `sha256:${'d'.repeat(64)}`;
const OUTPUT_ID = `sha256:${'e'.repeat(64)}`;

async function writeJson(filePath, value) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function externalReceipt(role, overrides = {}) {
  const source = role === 'source';
  return {
    schema: 'doppler.runtime-ownership-execution-evidence/v1',
    role,
    providerId: source ? 'huggingface-transformers-pytorch' : 'transformersjs-webgpu',
    artifactId: source
      ? 'Qwen/Qwen3.5-0.8B'
      : 'onnx-community/Qwen3.5-0.8B-ONNX@q4f16',
    artifactRevision: `${role}-fixture-revision`,
    workload: 'generation',
    logicalModelId: 'qwen-3-5-0-8b-q4k-ehaf16',
    runtime: {
      name: `${role}-runtime`,
      version: '1.0.0-fixture',
      backendId: `${role}-backend`,
      environmentFingerprint: ENVIRONMENT_ID,
    },
    invocation: { configurationDigest: CONFIGURATION_ID },
    result: {
      status: 'passed',
      outputDigest: OUTPUT_ID,
      startedAtUtc: '2026-08-15T19:00:00.000Z',
      completedAtUtc: '2026-08-15T19:01:00.000Z',
    },
    ...overrides,
  };
}

function dopplerReceipt() {
  return {
    receiptVersion: 'doppler_provider_receipt_v1',
    receiptId: 'generation-doppler-receipt',
    source: 'local',
    model: {
      id: 'qwen-3-5-0-8b-q4k-ehaf16',
      hash: ARTIFACT_ID,
    },
    device: { vendor: 'fixture-vendor' },
    failure: null,
    fallbackDecision: null,
    timestamp: '2026-08-15T19:01:00.000Z',
    resolutionStatus: 'resolved',
    resolution: {
      schema: 'doppler.resolution-identity/v1',
      logicalModelId: 'qwen-3-5-0-8b-q4k-ehaf16',
      resolvedArtifactVariantId: ARTIFACT_ID,
      resolvedExecutionId: EXECUTION_ID,
    },
  };
}

function capture(overrides = {}) {
  return {
    schema: 'doppler.runtime-ownership-evaluation-capture/v1',
    decisionId: 'qwen35-generation-runtime-ownership',
    recommendedDisposition: 'doppler',
    decisionRationale: 'The frozen material-advantage threshold passed with controls intact.',
    qualifiedAtUtc: '2026-08-15T21:00:00.000Z',
    expiresAtUtc: '2026-09-14T21:00:00.000Z',
    hypothesisResults: [
      {
        axis: 'end-to-end-performance',
        passed: true,
        observedValue: 1.5,
        evaluatedAtUtc: '2026-08-15T20:00:00.000Z',
        evidencePath: 'evidence/end-to-end-performance.json',
      },
    ],
    evidence: {
      sourceExecution: 'evidence/source-execution.json',
      incumbentExecution: 'evidence/incumbent-execution.json',
      dopplerExecution: 'evidence/doppler-execution.json',
      correctness: 'evidence/correctness.json',
      taskQuality: 'evidence/task-quality.json',
      usability: 'evidence/usability.json',
      memory: 'evidence/memory.json',
      endToEndPerformance: 'evidence/end-to-end-performance.json',
      diagnosticDepth: 'evidence/diagnostic-depth.json',
      distributionCost: 'evidence/distribution-cost.json',
      integrationBurden: 'evidence/integration-burden.json',
      providerRisk: 'evidence/provider-risk.json',
    },
    ...overrides,
  };
}

async function createFixture() {
  const repoRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-ownership-record-'));
  const policy = JSON.parse(await fs.readFile(
    path.join(SOURCE_ROOT, 'benchmarks/vendors/runtime-ownership-decisions.json'),
    'utf8'
  ));
  const policyPath = path.join(repoRoot, 'benchmarks/vendors/runtime-ownership-decisions.json');
  const capturePath = path.join(repoRoot, 'capture.json');
  const outputPolicyPath = path.join(repoRoot, 'runtime-ownership.next.json');
  await writeJson(policyPath, policy);
  for (const evidencePath of Object.values(capture().evidence)) {
    await writeJson(path.join(repoRoot, evidencePath), { fixture: true });
  }
  await writeJson(
    path.join(repoRoot, capture().evidence.sourceExecution),
    externalReceipt('source')
  );
  await writeJson(
    path.join(repoRoot, capture().evidence.incumbentExecution),
    externalReceipt('incumbent')
  );
  await writeJson(
    path.join(repoRoot, capture().evidence.dopplerExecution),
    dopplerReceipt()
  );
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
  const result = await recordRuntimeOwnershipEvaluation({ ...fixture, now: NOW });
  assert.equal(result.claimAllowed, false);
  assert.equal(result.disposition, 'doppler');
  assert.equal(result.resolvedArtifactVariantId, ARTIFACT_ID);
  assert.equal(result.resolvedExecutionId, EXECUTION_ID);
  assert.match(result.sourceExecutionId, /^sha256:[0-9a-f]{64}$/);
  assert.match(result.incumbentExecutionId, /^sha256:[0-9a-f]{64}$/);
  assert.deepEqual(result.blockers, [
    'runtime-ownership-evaluation-awaiting-explicit-promotion',
  ]);
  const output = JSON.parse(await fs.readFile(fixture.outputPolicyPath, 'utf8'));
  const decision = output.decisions.find((entry) => entry.id === capture().decisionId);
  assert.equal(decision.claimAllowed, false);
  assert.equal(decision.hypotheses[0].threshold.value, 1.25);
  assert.equal(decision.hypotheses[0].result.observedValue, 1.5);
});

await withFixture(async (fixture) => {
  const sourcePath = path.join(fixture.repoRoot, capture().evidence.sourceExecution);
  await writeJson(sourcePath, externalReceipt('source', { providerId: 'different-provider' }));
  await assert.rejects(
    () => recordRuntimeOwnershipEvaluation({ ...fixture, now: NOW }),
    /source execution evidence is invalid.*does not match expected/
  );
  await assert.rejects(() => fs.stat(fixture.outputPolicyPath));
});

await withFixture(async (fixture) => {
  await writeJson(fixture.capturePath, capture({ hypothesisResults: [] }));
  await assert.rejects(
    () => recordRuntimeOwnershipEvaluation({ ...fixture, now: NOW }),
    /Missing hypothesis result axis end-to-end-performance/
  );
});

await withFixture(async (fixture) => {
  await writeJson(fixture.capturePath, capture({ recommendedDisposition: 'incumbent' }));
  await assert.rejects(
    () => recordRuntimeOwnershipEvaluation({ ...fixture, now: NOW }),
    /incumbent recommendation conflicts with a passing material advantage/
  );
});

await withFixture(async (fixture) => {
  const sourcePath = path.join(fixture.repoRoot, capture().evidence.sourceExecution);
  await writeJson(sourcePath, externalReceipt('source', {
    result: {
      status: 'failed',
      outputDigest: null,
      startedAtUtc: '2026-08-15T19:00:00.000Z',
      completedAtUtc: '2026-08-15T19:01:00.000Z',
    },
  }));
  await writeJson(fixture.capturePath, capture({
    recommendedDisposition: 'incumbent',
    decisionRationale: 'The material-advantage threshold did not pass.',
    hypothesisResults: [
      {
        ...capture().hypothesisResults[0],
        passed: false,
        observedValue: 1,
      },
    ],
  }));
  const result = await recordRuntimeOwnershipEvaluation({ ...fixture, now: NOW });
  assert.equal(result.claimAllowed, false);
  assert.ok(result.blockers.includes('source-execution-not-passed'));
});

await withFixture(async (fixture) => {
  await recordRuntimeOwnershipEvaluation({ ...fixture, now: NOW });
  await fs.copyFile(fixture.outputPolicyPath, fixture.policyPath);
  await assert.rejects(
    () => recordRuntimeOwnershipEvaluation({ ...fixture, now: NOW }),
    /already contains evaluation state/
  );
  const result = await recordRuntimeOwnershipEvaluation({
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
    'benchmarks/vendors/runtime-ownership-decisions.json',
  ]),
  /explicit --apply/
);
assert.equal(
  parseArgs(['--capture', 'capture.json', '--apply']).outputPolicyPath,
  path.join(SOURCE_ROOT, 'benchmarks/vendors/runtime-ownership-decisions.json')
);

console.log('record-runtime-ownership-evaluation.test: ok');
