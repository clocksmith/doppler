import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import {
  buildProviderConformanceReport,
  validateProviderConformancePolicy,
} from '../../tools/check-provider-conformance.js';
import { computeCanonicalJsonSha256 } from '../../tools/lib/canonical-json.js';
import {
  PROVIDER_CONFORMANCE_EVIDENCE_CLASSES,
  computeProviderConformanceEvidenceSetDigest,
  computeProviderConformanceProviderSetDigest,
} from '../../tools/lib/provider-conformance-evidence.js';

const POLICY_PATH = path.join(process.cwd(), 'tools/policies/provider-conformance.json');
const NOW = new Date('2026-08-15T12:00:00.000Z');
const ARTIFACT_ID = `sha256:${'a'.repeat(64)}`;
const TOKENIZER_ID = `sha256:${'b'.repeat(64)}`;
const GRAPH_ID = `sha256:${'c'.repeat(64)}`;
const RUNTIME_POLICY_ID = `sha256:${'d'.repeat(64)}`;
const OUTPUT_ID = `sha256:${'e'.repeat(64)}`;
const HELD_OUT_ID = `sha256:${'f'.repeat(64)}`;
const HARNESS_REVISION = '1'.repeat(40);
const TEST_ROOT = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-provider-conformance-'));

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

async function writeJson(relativePath, value) {
  const filePath = path.join(TEST_ROOT, relativePath);
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
  return { path: relativePath, digest: computeCanonicalJsonSha256(value) };
}

async function writeText(relativePath) {
  const filePath = path.join(TEST_ROOT, relativePath);
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, 'fixture\n', 'utf8');
}

function executionId(suite, laneId) {
  return computeCanonicalJsonSha256({ suiteId: suite.id, laneId, kind: 'execution' });
}

function environmentId(suite, laneId) {
  return computeCanonicalJsonSha256({ suiteId: suite.id, laneId, kind: 'environment' });
}

function executionReceipt(suite, laneId) {
  return {
    receiptVersion: 'doppler_provider_receipt_v1',
    receiptId: `${suite.id}-${laneId}-execution`,
    source: 'local',
    model: { id: suite.logicalModelId, hash: ARTIFACT_ID },
    device: { vendor: 'fixture-vendor', laneId },
    failure: null,
    fallbackDecision: null,
    timestamp: '2026-08-01T00:00:00.000Z',
    resolutionStatus: 'resolved',
    resolution: {
      schema: 'doppler.resolution-identity/v1',
      logicalModelId: suite.logicalModelId,
      resolvedArtifactVariantId: ARTIFACT_ID,
      resolvedExecutionId: executionId(suite, laneId),
    },
  };
}

function observationsFor(suite, evidenceClass) {
  if (evidenceClass === 'modelContract') {
    return {
      manifestDigest: ARTIFACT_ID,
      tokenizerDigest: TOKENIZER_ID,
      executionGraphDigest: GRAPH_ID,
      runtimePolicyDigest: RUNTIME_POLICY_ID,
      artifactValidated: true,
      tokenizerIdentityMatched: true,
      executionGraphIdentityMatched: true,
      runtimePolicyExplicit: true,
    };
  }
  if (evidenceClass === 'resolutionIdentity') {
    return {
      logicalModelResolved: true,
      manifestVariantMatched: true,
      artifactDigestMatched: true,
      executionDigestMatched: true,
      fallbackUsed: false,
    };
  }
  if (evidenceClass === 'operations') {
    return {
      declaredOperations: suite.declaredOperations,
      observedOperations: suite.declaredOperations,
      unsupportedOperationUsed: false,
    };
  }
  if (evidenceClass === 'lifecycle') {
    return {
      load: 'passed',
      execute: 'passed',
      unload: 'passed',
      repeatedSessions: 5,
      minimumRepeatedSessions: 3,
    };
  }
  if (suite.correctnessClass === 'exact-token') {
    return {
      correctnessClass: suite.correctnessClass,
      referenceOutputDigest: OUTPUT_ID,
      providerOutputDigest: OUTPUT_ID,
      tokenParityPassed: true,
      deterministicContinuationPassed: true,
    };
  }
  if (suite.correctnessClass === 'tolerance-bounded-numerical') {
    return {
      correctnessClass: suite.correctnessClass,
      referenceOutputDigest: OUTPUT_ID,
      providerOutputDigest: computeCanonicalJsonSha256({ output: suite.id }),
      shapeMatched: true,
      finiteOutputs: true,
      maxAbsoluteError: 0.0001,
      maximumAbsoluteError: 0.001,
    };
  }
  return {
    correctnessClass: suite.correctnessClass,
    heldOutSetDigest: HELD_OUT_ID,
    referenceScore: 0.95,
    providerScore: 0.94,
    minimumAcceptedScore: 0.9,
    maximumReferenceDelta: 0.02,
    higherIsBetter: true,
    orderingAgreementPassed: true,
  };
}

function semanticReceipt(suite, laneId, evidenceClass, providerReceiptDigest) {
  return {
    schema: 'doppler.provider-conformance-evidence/v1',
    evidenceClass,
    suiteId: suite.id,
    laneId,
    workload: suite.workload,
    logicalModelId: suite.logicalModelId,
    manifestVariantId: suite.manifestVariantId,
    resolvedArtifactVariantId: ARTIFACT_ID,
    resolvedExecutionId: executionId(suite, laneId),
    implementationId: `${laneId}-implementation-v1`,
    harnessRevision: HARNESS_REVISION,
    environmentFingerprint: environmentId(suite, laneId),
    providerReceiptDigest,
    capturedAtUtc: '2026-08-01T00:15:00.000Z',
    result: { passed: true },
    observations: observationsFor(suite, evidenceClass),
  };
}

async function qualifiedProvider(suite, laneId) {
  const providerReceipt = await writeJson(
    `evidence/${suite.id}-${laneId}-provider.json`,
    executionReceipt(suite, laneId)
  );
  const evidence = { providerReceipt };
  for (const evidenceClass of PROVIDER_CONFORMANCE_EVIDENCE_CLASSES) {
    evidence[evidenceClass] = await writeJson(
      `evidence/${suite.id}-${laneId}-${evidenceClass}.json`,
      semanticReceipt(suite, laneId, evidenceClass, providerReceipt.digest)
    );
  }
  const evidenceSet = Object.fromEntries([
    ...PROVIDER_CONFORMANCE_EVIDENCE_CLASSES,
    'providerReceipt',
  ].map((field) => [field, evidence[field]]));
  evidence.promotion = await writeJson(
    `evidence/${suite.id}-${laneId}-promotion.json`,
    {
      schema: 'doppler.provider-conformance-provider-promotion/v1',
      suiteId: suite.id,
      laneId,
      logicalModelId: suite.logicalModelId,
      resolvedArtifactVariantId: ARTIFACT_ID,
      resolvedExecutionId: executionId(suite, laneId),
      implementationId: `${laneId}-implementation-v1`,
      evidenceSetDigest: computeProviderConformanceEvidenceSetDigest(evidenceSet),
      decision: 'promote',
      authority: 'human',
      reviewer: 'provider-reviewer',
      rationale: 'Exact tuple satisfied the declared semantic gates.',
      promotedAtUtc: '2026-08-01T02:00:00.000Z',
      qualifiedAtUtc: '2026-08-01T01:00:00.000Z',
      expiresAtUtc: '2026-10-01T00:00:00.000Z',
    }
  );
  return {
    laneId,
    implementationId: `${laneId}-implementation-v1`,
    harnessRevision: HARNESS_REVISION,
    logicalModelId: suite.logicalModelId,
    manifestVariantId: suite.manifestVariantId,
    resolvedArtifactVariantId: ARTIFACT_ID,
    resolvedExecutionId: executionId(suite, laneId),
    environmentFingerprint: environmentId(suite, laneId),
    operations: suite.declaredOperations,
    lifecycle: { load: 'passed', execute: 'passed', unload: 'passed' },
    correctness: { class: suite.correctnessClass, passed: true },
    qualifiedAtUtc: '2026-08-01T01:00:00.000Z',
    expiresAtUtc: '2026-10-01T00:00:00.000Z',
    evidence,
    claimAllowed: true,
    blockers: [],
  };
}

async function qualifiedSuite(template) {
  const suite = clone(template);
  suite.resolvedArtifactVariantId = ARTIFACT_ID;
  suite.providers = await Promise.all(
    suite.requiredProviderLaneIds.map((laneId) => qualifiedProvider(suite, laneId))
  );
  const providerSet = suite.providers.map((provider) => {
    const evidenceSet = Object.fromEntries([
      ...PROVIDER_CONFORMANCE_EVIDENCE_CLASSES,
      'providerReceipt',
    ].map((field) => [field, provider.evidence[field]]));
    return {
      laneId: provider.laneId,
      resolvedExecutionId: provider.resolvedExecutionId,
      evidenceSetDigest: computeProviderConformanceEvidenceSetDigest(evidenceSet),
      promotionDigest: provider.evidence.promotion.digest,
      qualifiedAtUtc: provider.qualifiedAtUtc,
      expiresAtUtc: provider.expiresAtUtc,
    };
  });
  suite.promotion = await writeJson(`evidence/${suite.id}-suite-promotion.json`, {
    schema: 'doppler.provider-conformance-suite-promotion/v1',
    suiteId: suite.id,
    workload: suite.workload,
    logicalModelId: suite.logicalModelId,
    resolvedArtifactVariantId: ARTIFACT_ID,
    requiredProviderLaneIds: suite.requiredProviderLaneIds,
    providerSetDigest: computeProviderConformanceProviderSetDigest(providerSet),
    decision: 'promote',
    authority: 'human',
    reviewer: 'suite-reviewer',
    rationale: 'Both required provider lanes passed the same exact workload tuple.',
    promotedAtUtc: '2026-08-01T03:00:00.000Z',
    expiresAtUtc: '2026-10-01T00:00:00.000Z',
  });
  suite.claimAllowed = true;
  suite.blockers = [];
  return suite;
}

const policy = JSON.parse(await fs.readFile(POLICY_PATH, 'utf8'));
for (const lane of policy.providerLanes) await writeText(lane.contractPath);
for (const suite of policy.suites) await writeText(suite.workloadContractPath);

{
  const report = await buildProviderConformanceReport({ policyPath: POLICY_PATH, now: NOW });
  assert.equal(report.ok, true, report.errors.join('\n'));
  assert.equal(report.gateSatisfied, false);
  assert.equal(report.qualifiedSuites, 0);
  assert.equal(report.candidateSuites, 3);
}

{
  const complete = clone(policy);
  complete.suites = await Promise.all(policy.suites.map(qualifiedSuite));
  const report = await validateProviderConformancePolicy(complete, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.deepEqual(report.errors, []);
  assert.equal(report.qualifiedSuites, 3);
  assert.equal(report.gateSatisfied, true);
}

{
  const noPromotion = clone(policy);
  const suite = await qualifiedSuite(policy.suites[0]);
  suite.providers[0].evidence.promotion = null;
  suite.providers[0].claimAllowed = true;
  suite.promotion = null;
  noPromotion.suites = [suite];
  const report = await validateProviderConformancePolicy(noPromotion, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(
    report.errors.some((error) => error.includes('claimAllowed record does not satisfy')),
    report.errors.join('\n')
  );
}

{
  const arbitraryFixture = clone(policy);
  const suite = await qualifiedSuite(policy.suites[0]);
  const fixture = await writeJson('evidence/arbitrary.json', { fixture: true });
  suite.providers[0].evidence.modelContract = fixture;
  arbitraryFixture.suites = [suite];
  const report = await validateProviderConformancePolicy(arbitraryFixture, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(
    report.errors.some((error) => error.includes('evidence.modelContract: provider conformance evidence.schema is required')),
    report.errors.join('\n')
  );
}

{
  const falseResult = clone(policy);
  const suite = await qualifiedSuite(policy.suites[0]);
  const lifecyclePath = suite.providers[0].evidence.lifecycle.path;
  const receipt = JSON.parse(await fs.readFile(path.join(TEST_ROOT, lifecyclePath), 'utf8'));
  receipt.observations.unload = 'failed';
  suite.providers[0].evidence.lifecycle = await writeJson(lifecyclePath, receipt);
  falseResult.suites = [suite];
  const report = await validateProviderConformancePolicy(falseResult, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(
    report.errors.some((error) => error.includes('result.passed does not match observations')),
    report.errors.join('\n')
  );
}

{
  const tampered = clone(policy);
  const suite = await qualifiedSuite(policy.suites[1]);
  const pathToReceipt = suite.providers[0].evidence.correctness.path;
  const receipt = JSON.parse(await fs.readFile(path.join(TEST_ROOT, pathToReceipt), 'utf8'));
  receipt.observations.maxAbsoluteError = 1;
  await fs.writeFile(path.join(TEST_ROOT, pathToReceipt), `${JSON.stringify(receipt, null, 2)}\n`);
  tampered.suites = [suite];
  const report = await validateProviderConformancePolicy(tampered, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(
    report.errors.some((error) => error.includes('digest does not match canonical JSON evidence')),
    report.errors.join('\n')
  );
}

{
  const providerSetDrift = clone(policy);
  const suite = await qualifiedSuite(policy.suites[2]);
  suite.providers[1].qualifiedAtUtc = '2026-08-01T01:30:00.000Z';
  providerSetDrift.suites = [suite];
  const report = await validateProviderConformancePolicy(providerSetDrift, {
    repoRoot: TEST_ROOT,
    now: NOW,
  });
  assert.ok(
    report.errors.some((error) => error.includes('providerSetDigest does not match')),
    report.errors.join('\n')
  );
}

await fs.rm(TEST_ROOT, { recursive: true, force: true });

console.log('provider-conformance.test: ok');
