import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import {
  buildRuntimeOwnershipDecisionReport,
  validateRuntimeOwnershipDecisions,
} from '../../tools/check-runtime-ownership-decisions.js';
import {
  computeRuntimeOwnershipEvidenceId,
} from '../../tools/lib/runtime-ownership-execution-evidence.js';

const POLICY_PATH = path.join(
  process.cwd(),
  'benchmarks',
  'vendors',
  'runtime-ownership-decisions.json'
);
const NOW = new Date('2026-08-15T12:00:00.000Z');
const ARTIFACT_ID = `sha256:${'a'.repeat(64)}`;
const EXECUTION_ID = `sha256:${'b'.repeat(64)}`;
const ENVIRONMENT_ID = `sha256:${'c'.repeat(64)}`;
const CONFIGURATION_ID = `sha256:${'d'.repeat(64)}`;
const OUTPUT_ID = `sha256:${'e'.repeat(64)}`;
const TEST_ROOT = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-runtime-ownership-'));
const EVIDENCE_ROOT = path.join(TEST_ROOT, 'evidence');

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

async function writeJson(relativePath, value) {
  const filePath = path.join(TEST_ROOT, relativePath);
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function hypothesis(workload, passed = true) {
  return {
    axis: 'end-to-end-performance',
    statement: `Doppler materially improves ${workload} response time without quality loss.`,
    metric: 'doppler-over-incumbent-throughput-ratio',
    controlMetric: 'held-out-task-quality-delta',
    controlRequirement: 'quality delta remains within the frozen acceptance bound',
    threshold: {
      operator: 'greater-than-or-equal',
      value: 1.25,
      unit: 'ratio',
    },
    declaredAtUtc: '2026-07-15T00:00:00.000Z',
    result: {
      passed,
      observedValue: passed ? 1.5 : 1.0,
      evaluatedAtUtc: '2026-08-01T00:00:00.000Z',
      evidencePath: 'evidence/shared.json',
    },
  };
}

function decision(workload, disposition = 'doppler') {
  const evidence = Object.fromEntries([
    'sourceExecution',
    'incumbentExecution',
    'dopplerExecution',
    'correctness',
    'taskQuality',
    'usability',
    'memory',
    'endToEndPerformance',
    'diagnosticDepth',
    'distributionCost',
    'integrationBurden',
    'providerRisk',
  ].map((field) => [field, 'evidence/shared.json']));
  return {
    id: `${workload}-runtime-ownership`,
    workload,
    logicalModelId: `${workload}-logical-model`,
    manifestVariantId: `${workload}-manifest-variant`,
    resolvedArtifactVariantId: ARTIFACT_ID,
    resolvedExecutionId: EXECUTION_ID,
    sourceProviderId: 'authoritative-source-runtime',
    sourceArtifactId: `${workload}-source-artifact`,
    sourceExecutionId: null,
    incumbentProviderId: 'instrumented-incumbent',
    incumbentArtifactId: `${workload}-incumbent-artifact`,
    incumbentExecutionId: null,
    correctnessClass: workload === 'embedding'
      ? 'tolerance-bounded-numerical'
      : 'exact-token',
    hypotheses: [hypothesis(workload, disposition !== 'incumbent')],
    disposition,
    decisionRationale: `${disposition} is selected by the frozen material-advantage gate.`,
    qualifiedAtUtc: '2026-08-01T00:00:00.000Z',
    expiresAtUtc: '2026-10-01T00:00:00.000Z',
    evidence,
    claimAllowed: true,
    blockers: [],
  };
}

function externalExecutionEvidence(record, role) {
  const source = role === 'source';
  return {
    schema: 'doppler.runtime-ownership-execution-evidence/v1',
    role,
    providerId: source ? record.sourceProviderId : record.incumbentProviderId,
    artifactId: source ? record.sourceArtifactId : record.incumbentArtifactId,
    artifactRevision: `${record.workload}-${role}-revision`,
    workload: record.workload,
    logicalModelId: record.logicalModelId,
    runtime: {
      name: `${role}-runtime`,
      version: '1.0.0-fixture',
      backendId: `${role}-backend`,
      environmentFingerprint: ENVIRONMENT_ID,
    },
    invocation: {
      configurationDigest: CONFIGURATION_ID,
    },
    result: {
      status: 'passed',
      outputDigest: OUTPUT_ID,
      startedAtUtc: '2026-08-01T00:00:00.000Z',
      completedAtUtc: '2026-08-01T00:01:00.000Z',
    },
  };
}

function dopplerExecutionReceipt(record) {
  return {
    receiptVersion: 'doppler_provider_receipt_v1',
    receiptId: `${record.workload}-doppler-receipt`,
    source: 'local',
    model: { id: record.logicalModelId, hash: record.resolvedArtifactVariantId },
    device: { vendor: 'fixture' },
    failure: null,
    fallbackDecision: null,
    timestamp: '2026-08-01T00:01:00.000Z',
    resolutionStatus: 'resolved',
    resolution: {
      schema: 'doppler.resolution-identity/v1',
      logicalModelId: record.logicalModelId,
      resolvedArtifactVariantId: record.resolvedArtifactVariantId,
      resolvedExecutionId: record.resolvedExecutionId,
    },
  };
}

async function preparedDecision(workload, disposition = 'doppler') {
  const record = decision(workload, disposition);
  const source = externalExecutionEvidence(record, 'source');
  const incumbent = externalExecutionEvidence(record, 'incumbent');
  const sourcePath = `evidence/${workload}-source-execution.json`;
  const incumbentPath = `evidence/${workload}-incumbent-execution.json`;
  const dopplerPath = `evidence/${workload}-doppler-execution.json`;
  await writeJson(sourcePath, source);
  await writeJson(incumbentPath, incumbent);
  await writeJson(dopplerPath, dopplerExecutionReceipt(record));
  record.sourceExecutionId = computeRuntimeOwnershipEvidenceId(source);
  record.incumbentExecutionId = computeRuntimeOwnershipEvidenceId(incumbent);
  record.evidence.sourceExecution = sourcePath;
  record.evidence.incumbentExecution = incumbentPath;
  record.evidence.dopplerExecution = dopplerPath;
  return record;
}

const policy = JSON.parse(await fs.readFile(POLICY_PATH, 'utf8'));
await fs.mkdir(EVIDENCE_ROOT, { recursive: true });
await writeJson('evidence/shared.json', { fixture: true });

{
  const report = await buildRuntimeOwnershipDecisionReport({ policyPath: POLICY_PATH, now: NOW });
  assert.equal(report.ok, true, report.errors.join('\n'));
  assert.equal(report.gateSatisfied, false);
  assert.equal(report.qualifiedDecisions, 0);
  assert.equal(report.candidateDecisions, 3);
  assert.deepEqual(report.candidateWorkloads, ['generation', 'embedding', 'reranking']);
  assert.deepEqual(report.decisions.map((entry) => entry.id), [
    'qwen35-generation-runtime-ownership',
    'embeddinggemma-runtime-ownership',
    'qwen3-reranking-runtime-ownership',
  ]);
  assert.ok(report.decisions.every((entry) => entry.qualified === false));
  assert.ok(report.decisions.every((entry) => entry.hypothesisAxes.length === 1));
  assert.deepEqual(report.missingWorkloads, ['generation', 'embedding', 'reranking']);
}

{
  const complete = clone(policy);
  complete.decisions = await Promise.all([
    preparedDecision('generation', 'doppler'),
    preparedDecision('embedding', 'dual'),
    preparedDecision('reranking', 'incumbent'),
  ]);
  const report = await validateRuntimeOwnershipDecisions(complete, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.deepEqual(report.errors, []);
  assert.equal(report.qualifiedDecisions, 3);
  assert.equal(report.gateSatisfied, true);
}

{
  const unsupportedDoppler = clone(policy);
  const record = await preparedDecision('generation', 'doppler');
  record.hypotheses = [hypothesis('generation', false)];
  unsupportedDoppler.decisions = [record];
  const report = await validateRuntimeOwnershipDecisions(unsupportedDoppler, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(
    report.decisions[0].reasons.includes('doppler-disposition-without-material-advantage'),
    JSON.stringify(report.decisions[0], null, 2)
  );
  assert.ok(
    report.errors.includes(
      'generation-runtime-ownership: claimAllowed decision does not satisfy runtime ownership qualification'
    ),
    report.errors.join('\n')
  );
}

{
  const thresholdDrift = clone(policy);
  const record = await preparedDecision('embedding', 'doppler');
  record.hypotheses[0].result.passed = false;
  record.hypotheses[0].result.observedValue = 1.5;
  thresholdDrift.decisions = [record];
  const report = await validateRuntimeOwnershipDecisions(thresholdDrift, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(
    report.errors.some((error) => error.includes('passed does not match the declared threshold')),
    report.errors.join('\n')
  );
}

{
  const retrospective = clone(policy);
  const record = await preparedDecision('reranking', 'doppler');
  record.hypotheses[0].declaredAtUtc = '2026-08-02T00:00:00.000Z';
  retrospective.decisions = [record];
  const report = await validateRuntimeOwnershipDecisions(retrospective, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(
    report.decisions[0].reasons.includes('hypothesis-results-incomplete'),
    JSON.stringify(report.decisions[0], null, 2)
  );
}

{
  const prematureQualification = clone(policy);
  const record = await preparedDecision('generation', 'doppler');
  record.qualifiedAtUtc = '2026-07-20T00:00:00.000Z';
  prematureQualification.decisions = [record];
  const report = await validateRuntimeOwnershipDecisions(prematureQualification, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(
    report.decisions[0].reasons.includes('qualification-predates-hypothesis-evidence'),
    JSON.stringify(report.decisions[0], null, 2)
  );
}

{
  const missingEvidence = clone(policy);
  const record = await preparedDecision('generation', 'dual');
  record.evidence.providerRisk = null;
  missingEvidence.decisions = [record];
  const report = await validateRuntimeOwnershipDecisions(missingEvidence, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(report.decisions[0].missingEvidence.includes('providerRisk'));
  assert.equal(report.decisions[0].qualified, false);
}

{
  const incumbentConflict = clone(policy);
  const record = await preparedDecision('embedding', 'incumbent');
  record.hypotheses = [hypothesis('embedding', true)];
  incumbentConflict.decisions = [record];
  const report = await validateRuntimeOwnershipDecisions(incumbentConflict, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(
    report.decisions[0].reasons.includes(
      'incumbent-disposition-conflicts-with-material-advantage'
    )
  );
}

{
  const conflated = clone(policy);
  conflated.decisions[0].resolvedArtifactVariantId = conflated.decisions[0].manifestVariantId;
  const report = await validateRuntimeOwnershipDecisions(conflated, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(
    report.errors.includes(
      'qwen35-generation-runtime-ownership.resolvedArtifactVariantId must be a SHA-256 identity or null'
    ),
    report.errors.join('\n')
  );
}

{
  const opaqueIdentity = clone(policy);
  opaqueIdentity.decisions[0].sourceExecutionId = 'source-run-latest';
  const report = await validateRuntimeOwnershipDecisions(opaqueIdentity, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(
    report.errors.includes(
      'qwen35-generation-runtime-ownership.sourceExecutionId must be a SHA-256 identity or null'
    ),
    report.errors.join('\n')
  );
}

{
  const tampered = clone(policy);
  const record = await preparedDecision('generation', 'doppler');
  const receiptPath = record.evidence.sourceExecution;
  const receipt = JSON.parse(await fs.readFile(path.join(TEST_ROOT, receiptPath), 'utf8'));
  receipt.result.outputDigest = `sha256:${'f'.repeat(64)}`;
  await writeJson(receiptPath, receipt);
  tampered.decisions = [record];
  const report = await validateRuntimeOwnershipDecisions(tampered, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(
    report.errors.includes(
      'generation-runtime-ownership.sourceExecutionId does not match canonical sourceExecution evidence identity'
    ),
    report.errors.join('\n')
  );
}

{
  const mismatchedDoppler = clone(policy);
  const record = await preparedDecision('reranking', 'doppler');
  const receiptPath = record.evidence.dopplerExecution;
  const receipt = JSON.parse(await fs.readFile(path.join(TEST_ROOT, receiptPath), 'utf8'));
  receipt.resolution.resolvedExecutionId = `sha256:${'f'.repeat(64)}`;
  await writeJson(receiptPath, receipt);
  mismatchedDoppler.decisions = [record];
  const report = await validateRuntimeOwnershipDecisions(mismatchedDoppler, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(
    report.errors.some((error) => error.includes('Doppler execution identity')),
    report.errors.join('\n')
  );
}

await fs.rm(TEST_ROOT, { recursive: true, force: true });

console.log('runtime-ownership-decisions.test: ok');
