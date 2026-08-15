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
import {
  computeRuntimeOwnershipDecisionEvidenceDigest,
} from '../../tools/lib/runtime-ownership-decision-evidence.js';

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
const HELD_OUT_ID = `sha256:${'f'.repeat(64)}`;
const HARNESS_REVISION = '1'.repeat(40);
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
      evidence: null,
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
  ].map((field) => [field, null]));
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
    qualifiedAtUtc: '2026-08-01T01:00:00.000Z',
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

function evidenceReference(relativePath, receipt) {
  return {
    path: relativePath,
    digest: computeRuntimeOwnershipDecisionEvidenceDigest(receipt),
  };
}

function dimensionObservations(evidenceClass) {
  const observations = {
    correctness: {
      referenceValid: true,
      workloadEquivalent: true,
      incumbentAcceptable: true,
      dopplerAcceptable: true,
    },
    taskQuality: {
      heldOutSetDigest: HELD_OUT_ID,
      sourceScore: 0.9,
      incumbentScore: 0.88,
      dopplerScore: 0.89,
      minimumAcceptedScore: 0.85,
      higherIsBetter: true,
    },
    usability: {
      installSucceeded: true,
      loadSucceeded: true,
      invokeSucceeded: true,
      fallbackExplicit: true,
    },
    memory: {
      sourcePeakBytes: 500,
      incumbentPeakBytes: 450,
      dopplerPeakBytes: 400,
      maximumDopplerBytes: 425,
      measurementScopeMatched: true,
    },
    endToEndPerformance: {
      sourceValue: 100,
      incumbentValue: 80,
      dopplerValue: 120,
      unit: 'tokens-per-second',
      sampleCount: 10,
      minimumSampleCount: 5,
      timingScopeMatched: true,
      workloadEquivalent: true,
    },
    diagnosticDepth: {
      firstDivergenceLocalized: true,
      semanticBoundaryRecorded: true,
      correctionPathActionable: true,
      fallbackVisible: true,
    },
    distributionCost: {
      sourceArtifactBytes: 500,
      incumbentArtifactBytes: 450,
      dopplerArtifactBytes: 400,
      maximumDopplerBytes: 425,
      dependencyBytesCounted: true,
    },
    integrationBurden: {
      sourceSteps: 4,
      incumbentSteps: 3,
      dopplerSteps: 2,
      maximumDopplerSteps: 3,
      cleanInstallPassed: true,
      apiInvocationPassed: true,
    },
    providerRisk: {
      standardWebGpuPassed: true,
      selectedNodeProviderPassed: true,
      fallbackVisible: true,
      doeRequired: false,
      undeclaredProviderRequired: false,
    },
  };
  return observations[evidenceClass];
}

function boundEvidence(record) {
  return {
    decisionId: record.id,
    workload: record.workload,
    logicalModelId: record.logicalModelId,
    sourceExecutionId: record.sourceExecutionId,
    incumbentExecutionId: record.incumbentExecutionId,
    resolvedArtifactVariantId: record.resolvedArtifactVariantId,
    resolvedExecutionId: record.resolvedExecutionId,
    harnessRevision: HARNESS_REVISION,
    environmentFingerprint: ENVIRONMENT_ID,
  };
}

function dimensionEvidence(record, evidenceClass) {
  return {
    schema: 'doppler.runtime-ownership-dimension-evidence/v1',
    ...boundEvidence(record),
    evidenceClass,
    capturedAtUtc: '2026-08-01T00:00:00.000Z',
    result: {
      passed: true,
      observations: dimensionObservations(evidenceClass),
    },
  };
}

function hypothesisEvidence(record, passed) {
  const frozen = record.hypotheses[0];
  const qualitative = frozen.threshold.operator === 'pass';
  return {
    schema: 'doppler.runtime-ownership-hypothesis-evidence/v1',
    ...boundEvidence(record),
    axis: frozen.axis,
    metric: frozen.metric,
    controlMetric: frozen.controlMetric,
    evaluatedAtUtc: '2026-08-01T00:00:00.000Z',
    observedValue: qualitative ? (passed ? 'unsupported' : 'supported') : (passed ? 1.5 : 1),
    qualitativePassed: qualitative ? passed : null,
    controlPassed: true,
    endToEndAcceptancePassed: true,
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
  const doppler = dopplerExecutionReceipt(record);
  await writeJson(dopplerPath, doppler);
  record.sourceExecutionId = computeRuntimeOwnershipEvidenceId(source);
  record.incumbentExecutionId = computeRuntimeOwnershipEvidenceId(incumbent);
  record.evidence.sourceExecution = evidenceReference(sourcePath, source);
  record.evidence.incumbentExecution = evidenceReference(incumbentPath, incumbent);
  record.evidence.dopplerExecution = evidenceReference(dopplerPath, doppler);
  for (const evidenceClass of [
    'correctness',
    'taskQuality',
    'usability',
    'memory',
    'endToEndPerformance',
    'diagnosticDepth',
    'distributionCost',
    'integrationBurden',
    'providerRisk',
  ]) {
    const receipt = dimensionEvidence(record, evidenceClass);
    const receiptPath = `evidence/${workload}-${evidenceClass}.json`;
    await writeJson(receiptPath, receipt);
    record.evidence[evidenceClass] = evidenceReference(receiptPath, receipt);
  }
  const advantagePassed = disposition !== 'incumbent';
  const hypothesisReceipt = hypothesisEvidence(record, advantagePassed);
  const hypothesisPath = `evidence/${workload}-hypothesis.json`;
  await writeJson(hypothesisPath, hypothesisReceipt);
  record.hypotheses[0].result = {
    passed: advantagePassed,
    observedValue: advantagePassed ? 1.5 : 1,
    evaluatedAtUtc: hypothesisReceipt.evaluatedAtUtc,
    evidence: evidenceReference(hypothesisPath, hypothesisReceipt),
  };
  return record;
}

const policy = JSON.parse(await fs.readFile(POLICY_PATH, 'utf8'));
await fs.mkdir(EVIDENCE_ROOT, { recursive: true });

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
  const receipt = hypothesisEvidence(record, false);
  const reference = record.hypotheses[0].result.evidence;
  await writeJson(reference.path, receipt);
  record.hypotheses[0].result = {
    passed: false,
    observedValue: 1,
    evaluatedAtUtc: receipt.evaluatedAtUtc,
    evidence: evidenceReference(reference.path, receipt),
  };
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
    report.errors.some((error) => error.includes('passed does not match semantic hypothesis evidence')),
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
  const receipt = hypothesisEvidence(record, true);
  const reference = record.hypotheses[0].result.evidence;
  await writeJson(reference.path, receipt);
  record.hypotheses[0].result = {
    passed: true,
    observedValue: 1.5,
    evaluatedAtUtc: receipt.evaluatedAtUtc,
    evidence: evidenceReference(reference.path, receipt),
  };
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
  const unsupportedOperation = clone(policy);
  const record = await preparedDecision('generation', 'doppler');
  const incumbentPath = record.evidence.incumbentExecution.path;
  const incumbent = JSON.parse(await fs.readFile(path.join(TEST_ROOT, incumbentPath), 'utf8'));
  incumbent.result = {
    status: 'failed',
    outputDigest: null,
    startedAtUtc: '2026-08-01T00:00:00.000Z',
    completedAtUtc: '2026-08-01T00:01:00.000Z',
  };
  await writeJson(incumbentPath, incumbent);
  record.incumbentExecutionId = computeRuntimeOwnershipEvidenceId(incumbent);
  record.evidence.incumbentExecution = evidenceReference(incumbentPath, incumbent);
  record.hypotheses[0] = {
    axis: 'unsupported-operation',
    statement: 'The incumbent cannot execute the frozen workload while Doppler can.',
    metric: 'incumbent-operation-support',
    controlMetric: 'source-and-doppler-correctness',
    controlRequirement: 'Source and Doppler pass while the incumbent reports unsupported.',
    threshold: {
      operator: 'pass',
      value: null,
      unit: null,
    },
    declaredAtUtc: '2026-07-15T00:00:00.000Z',
    result: {
      passed: null,
      observedValue: null,
      evaluatedAtUtc: null,
      evidence: null,
    },
  };
  for (const evidenceClass of [
    'correctness',
    'taskQuality',
    'usability',
    'memory',
    'endToEndPerformance',
    'diagnosticDepth',
    'distributionCost',
    'integrationBurden',
    'providerRisk',
  ]) {
    const receipt = dimensionEvidence(record, evidenceClass);
    if (evidenceClass === 'correctness') {
      receipt.result.observations.incumbentAcceptable = false;
    }
    const receiptPath = record.evidence[evidenceClass].path;
    await writeJson(receiptPath, receipt);
    record.evidence[evidenceClass] = evidenceReference(receiptPath, receipt);
  }
  const hypothesisReceipt = hypothesisEvidence(record, true);
  const hypothesisPath = 'evidence/generation-unsupported-hypothesis.json';
  await writeJson(hypothesisPath, hypothesisReceipt);
  record.hypotheses[0].result = {
    passed: true,
    observedValue: 'unsupported',
    evaluatedAtUtc: hypothesisReceipt.evaluatedAtUtc,
    evidence: evidenceReference(hypothesisPath, hypothesisReceipt),
  };
  unsupportedOperation.decisions = [record];
  const report = await validateRuntimeOwnershipDecisions(unsupportedOperation, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.deepEqual(report.errors, []);
  assert.equal(report.decisions[0].qualified, true);
  assert.ok(!report.decisions[0].reasons.includes('incumbent-execution-not-passed'));
}

{
  const tampered = clone(policy);
  const record = await preparedDecision('generation', 'doppler');
  const receiptPath = record.evidence.sourceExecution.path;
  const receipt = JSON.parse(await fs.readFile(path.join(TEST_ROOT, receiptPath), 'utf8'));
  receipt.result.outputDigest = `sha256:${'f'.repeat(64)}`;
  await writeJson(receiptPath, receipt);
  tampered.decisions = [record];
  const report = await validateRuntimeOwnershipDecisions(tampered, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(
    report.errors.some((error) => error.includes(
      'generation-runtime-ownership.evidence.sourceExecution.digest does not match canonical JSON evidence'
    )),
    report.errors.join('\n')
  );
}

{
  const mismatchedDoppler = clone(policy);
  const record = await preparedDecision('reranking', 'doppler');
  const receiptPath = record.evidence.dopplerExecution.path;
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

{
  const arbitraryDimension = clone(policy);
  const record = await preparedDecision('generation', 'doppler');
  const receiptPath = record.evidence.providerRisk.path;
  const receipt = { fixture: true };
  await writeJson(receiptPath, receipt);
  record.evidence.providerRisk = evidenceReference(receiptPath, receipt);
  arbitraryDimension.decisions = [record];
  const report = await validateRuntimeOwnershipDecisions(arbitraryDimension, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(
    report.errors.some((error) => error.includes(
      'evidence.providerRisk: dimension evidence.schema is required'
    )),
    report.errors.join('\n')
  );
}

{
  const falseDimensionResult = clone(policy);
  const record = await preparedDecision('generation', 'doppler');
  const receiptPath = record.evidence.memory.path;
  const receipt = JSON.parse(await fs.readFile(path.join(TEST_ROOT, receiptPath), 'utf8'));
  receipt.result.observations.dopplerPeakBytes = 500;
  await writeJson(receiptPath, receipt);
  record.evidence.memory = evidenceReference(receiptPath, receipt);
  falseDimensionResult.decisions = [record];
  const report = await validateRuntimeOwnershipDecisions(falseDimensionResult, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(
    report.errors.some((error) => error.includes(
      'dimension evidence.result.passed does not match its observations'
    )),
    report.errors.join('\n')
  );
}

await fs.rm(TEST_ROOT, { recursive: true, force: true });

console.log('runtime-ownership-decisions.test: ok');
