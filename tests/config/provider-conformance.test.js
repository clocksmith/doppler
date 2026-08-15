import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';

import {
  buildProviderConformanceReport,
  validateProviderConformancePolicy,
} from '../../tools/check-provider-conformance.js';

const POLICY_PATH = path.join(process.cwd(), 'tools', 'policies', 'provider-conformance.json');
const NOW = new Date('2026-08-15T12:00:00.000Z');

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function provider(laneId, workload, modelId, artifactId) {
  return {
    laneId,
    implementationId: `${laneId}-implementation-v1`,
    logicalModelId: modelId,
    resolvedArtifactVariantId: artifactId,
    resolvedExecutionId: `${laneId}-${workload}-execution-v1`,
    environmentFingerprint: `${laneId}-environment-v1`,
    operations: [`${workload}-execute`],
    lifecycle: {
      load: 'passed',
      execute: 'passed',
      unload: 'passed',
    },
    correctness: {
      class: workload === 'embedding' ? 'tolerance-bounded-numerical' : 'exact-token',
      passed: true,
    },
    qualifiedAtUtc: '2026-08-01T00:00:00.000Z',
    expiresAtUtc: '2026-10-01T00:00:00.000Z',
    evidence: {
      modelContract: 'docs/goals.md',
      resolutionIdentity: 'docs/goals.md',
      operations: 'docs/goals.md',
      lifecycle: 'docs/goals.md',
      correctness: 'docs/goals.md',
      providerReceipt: 'docs/goals.md',
    },
    claimAllowed: true,
    blockers: [],
  };
}

function suite(workload) {
  const id = `${workload}-provider-conformance`;
  const modelId = `${workload}-logical-model`;
  const artifactId = `${workload}-artifact-variant`;
  return {
    id,
    workload,
    logicalModelId: modelId,
    resolvedArtifactVariantId: artifactId,
    workloadContractPath: 'docs/goals.md',
    declaredOperations: [`${workload}-execute`],
    correctnessClass: workload === 'embedding'
      ? 'tolerance-bounded-numerical'
      : 'exact-token',
    requiredProviderLaneIds: ['browser-webgpu', 'node-webgpu'],
    claimAllowed: true,
    providers: [
      provider('browser-webgpu', workload, modelId, artifactId),
      provider('node-webgpu', workload, modelId, artifactId),
    ],
    blockers: [],
  };
}

const policy = JSON.parse(await fs.readFile(POLICY_PATH, 'utf8'));

{
  const report = await buildProviderConformanceReport({ policyPath: POLICY_PATH, now: NOW });
  assert.equal(report.ok, true, report.errors.join('\n'));
  assert.equal(report.gateSatisfied, false);
  assert.equal(report.qualifiedSuites, 0);
  assert.deepEqual(report.missingWorkloads, ['generation', 'embedding', 'reranking']);
}

{
  const complete = clone(policy);
  complete.suites = ['generation', 'embedding', 'reranking'].map(suite);
  const report = await validateProviderConformancePolicy(complete, {
    repoRoot: process.cwd(),
    now: NOW,
  });
  assert.deepEqual(report.errors, []);
  assert.equal(report.qualifiedSuites, 3);
  assert.equal(report.gateSatisfied, true);
}

{
  const mismatched = clone(policy);
  mismatched.suites = [suite('generation')];
  mismatched.suites[0].providers[1].resolvedArtifactVariantId = 'different-artifact';
  const report = await validateProviderConformancePolicy(mismatched, {
    repoRoot: process.cwd(),
    now: NOW,
  });
  assert.ok(
    report.errors.includes(
      'generation-provider-conformance: provider node-webgpu: claimAllowed record does not satisfy provider conformance'
    ),
    report.errors.join('\n')
  );
  assert.equal(report.gateSatisfied, false);
}

{
  const incompleteLifecycle = clone(policy);
  incompleteLifecycle.suites = [suite('reranking')];
  incompleteLifecycle.suites[0].providers[0].lifecycle.unload = 'not-run';
  const report = await validateProviderConformancePolicy(incompleteLifecycle, {
    repoRoot: process.cwd(),
    now: NOW,
  });
  assert.ok(
    report.errors.some((error) => error.includes('provider browser-webgpu: claimAllowed record')),
    report.errors.join('\n')
  );
}

{
  const contractDrift = clone(policy);
  contractDrift.suites = [suite('generation')];
  contractDrift.suites[0].providers[1].operations = ['different-operation'];
  contractDrift.suites[0].providers[1].correctness.class = 'semantic';
  const report = await validateProviderConformancePolicy(contractDrift, {
    repoRoot: process.cwd(),
    now: NOW,
  });
  assert.ok(
    report.errors.some((error) => error.includes('provider node-webgpu: claimAllowed record')),
    report.errors.join('\n')
  );
  assert.deepEqual(
    report.suites[0].providers[1].reasons.filter((reason) => reason.endsWith('mismatch')),
    ['operations-mismatch', 'correctness-class-mismatch']
  );
}

{
  const missingEvidence = clone(policy);
  missingEvidence.suites = [suite('embedding')];
  missingEvidence.suites[0].providers[0].evidence.providerReceipt = null;
  const report = await validateProviderConformancePolicy(missingEvidence, {
    repoRoot: process.cwd(),
    now: NOW,
  });
  assert.ok(
    report.errors.some((error) => error.includes('provider browser-webgpu: claimAllowed record')),
    report.errors.join('\n')
  );
}

{
  const stale = clone(policy);
  stale.suites = [suite('reranking')];
  stale.suites[0].providers[0].qualifiedAtUtc = '2025-01-01T00:00:00.000Z';
  const report = await validateProviderConformancePolicy(stale, {
    repoRoot: process.cwd(),
    now: NOW,
  });
  assert.ok(
    report.suites[0].providers[0].reasons.includes('qualification-stale-or-future'),
    JSON.stringify(report.suites[0], null, 2)
  );
}

{
  const hiddenDoeRequirement = clone(policy);
  hiddenDoeRequirement.coreProviderLaneIds.push('doe');
  hiddenDoeRequirement.providerLanes.find((lane) => lane.id === 'doe').role = 'core';
  const report = await validateProviderConformancePolicy(hiddenDoeRequirement, {
    repoRoot: process.cwd(),
    now: NOW,
  });
  assert.ok(
    report.errors.includes(
      'provider conformance policy coreProviderLaneIds must be browser-webgpu, node-webgpu'
    ),
    report.errors.join('\n')
  );
  assert.ok(
    report.errors.includes('doe: Doe must remain an optional-named provider lane'),
    report.errors.join('\n')
  );
}

{
  const explicitDoe = clone(policy);
  const generation = suite('generation');
  generation.requiredProviderLaneIds.push('doe');
  generation.providers.push(provider(
    'doe',
    'generation',
    generation.logicalModelId,
    generation.resolvedArtifactVariantId
  ));
  explicitDoe.suites = [generation];
  const report = await validateProviderConformancePolicy(explicitDoe, {
    repoRoot: process.cwd(),
    now: NOW,
  });
  assert.deepEqual(report.errors, []);
  assert.equal(report.suites[0].qualified, true);
  assert.equal(report.gateSatisfied, false);
}

console.log('provider-conformance.test: ok');
