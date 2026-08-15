import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';

import {
  buildProductIntegrationQualificationReport,
  validateProductIntegrationQualification,
} from '../../tools/check-product-integration-qualification.js';

const POLICY_PATH = path.join(
  process.cwd(),
  'tools',
  'policies',
  'product-integration-qualification.json'
);
const NOW = new Date('2026-08-15T12:00:00.000Z');

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function integration(id, applicationName, workload) {
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
    resolvedArtifactVariantId: `${id}-artifact-variant`,
    resolvedExecutionId: `${id}-execution`,
    qualifiedAtUtc: '2026-08-01T00:00:00.000Z',
    expiresAtUtc: '2026-10-01T00:00:00.000Z',
    evidence: {
      installToFirstVerifiedOutput: 'docs/goals.md',
      identity: 'docs/goals.md',
      sourceTaskQualityRetention: 'docs/goals.md',
      reliability: 'docs/goals.md',
      memory: 'docs/goals.md',
      coldWarmResponse: 'docs/goals.md',
      browserHardwareQualification: 'docs/goals.md',
      incumbentControl: 'docs/goals.md',
      upgradeRequalification: 'docs/goals.md',
      rollbackRevocation: 'docs/goals.md',
    },
    blockers: [],
  };
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
  assert.deepEqual(report.integrations.map((entry) => entry.applicationName), [
    'Reploid',
    'Dream',
    'Columbo',
  ]);
  assert.ok(report.integrations.every((entry) => entry.qualified === false));
  assert.deepEqual(report.missingWorkloads, [
    'generation',
    'embedding-retrieval',
    'reranking',
  ]);
}

{
  const complete = clone(policy);
  complete.integrations = [
    integration('private-chat', 'Private Chat', 'generation'),
    integration('local-search', 'Local Search', 'embedding-retrieval'),
    integration('result-ranking', 'Result Ranking', 'reranking'),
  ];
  const report = await validateProductIntegrationQualification(complete, {
    repoRoot: process.cwd(),
    now: NOW,
  });
  assert.deepEqual(report.errors, []);
  assert.equal(report.qualifiedIntegrations, 3);
  assert.equal(report.distinctQualifiedApplications, 3);
  assert.equal(report.gateSatisfied, true);
}

{
  const stale = clone(policy);
  stale.integrations = [integration('private-chat', 'Private Chat', 'generation')];
  stale.integrations[0].ownerConfirmedAtUtc = '2025-01-01T00:00:00.000Z';
  const report = await validateProductIntegrationQualification(stale, {
    repoRoot: process.cwd(),
    now: NOW,
  });
  assert.ok(
    report.errors.includes(
      'private-chat: claimAllowed integration does not satisfy product qualification'
    ),
    report.errors.join('\n')
  );
  assert.equal(report.gateSatisfied, false);
}

{
  const candidate = clone(policy);
  const record = integration('private-chat', 'Private Chat', 'generation');
  record.qualificationLevel = 'runtime-verified';
  record.lifecycle = 'candidate';
  record.claimAllowed = false;
  record.blockers = ['held-out-task-gate-missing'];
  record.evidence.sourceTaskQualityRetention = null;
  candidate.integrations = [record];
  const report = await validateProductIntegrationQualification(candidate, {
    repoRoot: process.cwd(),
    now: NOW,
  });
  assert.deepEqual(report.errors, []);
  assert.equal(report.integrations[0].qualified, false);
  assert.equal(report.candidateIntegrations, 1);
  assert.ok(report.integrations[0].missingEvidence.includes('sourceTaskQualityRetention'));
}

console.log('product-integration-qualification.test: ok');
