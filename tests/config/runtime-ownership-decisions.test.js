import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';

import {
  buildRuntimeOwnershipDecisionReport,
  validateRuntimeOwnershipDecisions,
} from '../../tools/check-runtime-ownership-decisions.js';

const POLICY_PATH = path.join(
  process.cwd(),
  'benchmarks',
  'vendors',
  'runtime-ownership-decisions.json'
);
const NOW = new Date('2026-08-15T12:00:00.000Z');

function clone(value) {
  return JSON.parse(JSON.stringify(value));
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
      evidencePath: 'docs/goals.md',
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
  ].map((field) => [field, 'docs/goals.md']));
  return {
    id: `${workload}-runtime-ownership`,
    workload,
    logicalModelId: `${workload}-logical-model`,
    resolvedArtifactVariantId: `${workload}-artifact-variant`,
    resolvedExecutionId: `${workload}-execution`,
    sourceProviderId: 'authoritative-source-runtime',
    sourceArtifactId: `${workload}-source-artifact`,
    sourceExecutionId: `${workload}-source-execution`,
    incumbentProviderId: 'instrumented-incumbent',
    incumbentArtifactId: `${workload}-incumbent-artifact`,
    incumbentExecutionId: `${workload}-incumbent-execution`,
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

const policy = JSON.parse(await fs.readFile(POLICY_PATH, 'utf8'));

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
  complete.decisions = [
    decision('generation', 'doppler'),
    decision('embedding', 'dual'),
    decision('reranking', 'incumbent'),
  ];
  const report = await validateRuntimeOwnershipDecisions(complete, {
    repoRoot: process.cwd(),
    now: NOW,
  });
  assert.deepEqual(report.errors, []);
  assert.equal(report.qualifiedDecisions, 3);
  assert.equal(report.gateSatisfied, true);
}

{
  const unsupportedDoppler = clone(policy);
  const record = decision('generation', 'doppler');
  record.hypotheses = [hypothesis('generation', false)];
  unsupportedDoppler.decisions = [record];
  const report = await validateRuntimeOwnershipDecisions(unsupportedDoppler, {
    repoRoot: process.cwd(),
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
  const record = decision('embedding', 'doppler');
  record.hypotheses[0].result.passed = false;
  record.hypotheses[0].result.observedValue = 1.5;
  thresholdDrift.decisions = [record];
  const report = await validateRuntimeOwnershipDecisions(thresholdDrift, {
    repoRoot: process.cwd(),
    now: NOW,
  });
  assert.ok(
    report.errors.some((error) => error.includes('passed does not match the declared threshold')),
    report.errors.join('\n')
  );
}

{
  const retrospective = clone(policy);
  const record = decision('reranking', 'doppler');
  record.hypotheses[0].declaredAtUtc = '2026-08-02T00:00:00.000Z';
  retrospective.decisions = [record];
  const report = await validateRuntimeOwnershipDecisions(retrospective, {
    repoRoot: process.cwd(),
    now: NOW,
  });
  assert.ok(
    report.decisions[0].reasons.includes('hypothesis-results-incomplete'),
    JSON.stringify(report.decisions[0], null, 2)
  );
}

{
  const prematureQualification = clone(policy);
  const record = decision('generation', 'doppler');
  record.qualifiedAtUtc = '2026-07-20T00:00:00.000Z';
  prematureQualification.decisions = [record];
  const report = await validateRuntimeOwnershipDecisions(prematureQualification, {
    repoRoot: process.cwd(),
    now: NOW,
  });
  assert.ok(
    report.decisions[0].reasons.includes('qualification-predates-hypothesis-evidence'),
    JSON.stringify(report.decisions[0], null, 2)
  );
}

{
  const missingEvidence = clone(policy);
  const record = decision('generation', 'dual');
  record.evidence.providerRisk = null;
  missingEvidence.decisions = [record];
  const report = await validateRuntimeOwnershipDecisions(missingEvidence, {
    repoRoot: process.cwd(),
    now: NOW,
  });
  assert.ok(report.decisions[0].missingEvidence.includes('providerRisk'));
  assert.equal(report.decisions[0].qualified, false);
}

{
  const incumbentConflict = clone(policy);
  const record = decision('embedding', 'incumbent');
  record.hypotheses = [hypothesis('embedding', true)];
  incumbentConflict.decisions = [record];
  const report = await validateRuntimeOwnershipDecisions(incumbentConflict, {
    repoRoot: process.cwd(),
    now: NOW,
  });
  assert.ok(
    report.decisions[0].reasons.includes(
      'incumbent-disposition-conflicts-with-material-advantage'
    )
  );
}

console.log('runtime-ownership-decisions.test: ok');
