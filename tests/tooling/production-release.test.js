import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import {
  APPLICATION_GATE_RECEIPT_SCHEMA,
  hashProductionReleaseEvidence,
  signProductionReleaseEvidence,
  verifyProductionReleaseEvidenceSignature,
} from '../../src/config/production-release-evidence.js';
import { hashProductionRelease } from '../../src/config/production-release.js';
import { createPackReleaseFixture, createSignedPackFixture, TEST_PACK_AUTHORITY, TEST_PACK_PUBLIC_KEY } from '../helpers/pack-v2-fixture.js';
import { runProductionRelease } from '../../src/tooling/production-release.js';

const sha = (character) => `sha256:${character.repeat(64)}`;
const signingPrivate = {
  crv: 'Ed25519',
  d: 'WQi2FHRfw0jZxl_IXiMp5TAuehMfssojWd2Oj3WaUKU',
  x: 'FLU5-eSyW8ORkAf8HupzJn8juiJ2TrGSw2rgMNqGPfc',
  kty: 'OKP',
};
const signingPublic = { crv: signingPrivate.crv, x: signingPrivate.x, kty: signingPrivate.kty };
const tmpRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-production-release-'));
const packPath = path.join(tmpRoot, 'candidate.pack.json');
const releaseContract = createPackReleaseFixture({ targetIds: ['webgpu-f32-portable'] });
const { pack } = await createSignedPackFixture({ release: releaseContract });
await fs.writeFile(packPath, `${JSON.stringify(pack)}\n`, 'utf8');

const release = {
  schema: 'doppler.production-release/v1',
  schemaVersion: 1,
  releaseId: '',
  createdAtUtc: '2026-08-24T00:00:00.000Z',
  evidenceClass: 'reference-fixture',
  candidate: {
    logicalModelId: pack.modelId,
    sourceRevision: releaseContract.source.revision,
    sourceRevisionDigest: releaseContract.source.revisionDigest,
    packPath: 'candidate.pack.json',
    packSemanticRoot: pack.semanticRoot,
  },
  application: {
    applicationId: releaseContract.application.applicationId,
    platform: 'electron',
    revision: releaseContract.application.applicationRevision,
    revisionDigest: releaseContract.application.applicationRevisionDigest,
    rendererEntry: 'renderer.js',
    mainEntry: 'main.js',
  },
  acceptance: {
    workload: releaseContract.application.workload,
    oracle: releaseContract.application.oracle,
    tests: [{
      id: 'application-acceptance',
      command: ['node', 'acceptance.js'],
      workdir: '.',
      timeoutMs: 10000,
      evidenceSchema: APPLICATION_GATE_RECEIPT_SCHEMA,
    }],
    thresholds: {
      quality: { metric: 'ndcg-at-ten', minimum: 0.9 },
      coldLatencyMs: { maximum: 100 },
      warmLatencyMs: { maximum: 50 },
      peakMemoryBytes: { maximum: 1024 },
      failureRate: { maximum: 0 },
    },
    incumbentControl: {
      providerId: 'fixture-provider',
      artifactRevision: 'fixture-provider-v1',
      executionDigest: sha('6'),
    },
  },
  supportedDevices: {
    policyId: 'fixture-electron-fleet',
    policyDigest: sha('7'),
    receiptMode: 'customer-operated-agent',
    targets: [
      {
        id: 'windows-x64-webgpu', os: 'windows', osVersionRange: '>=10.0.22631',
        architectures: ['x64'], electronVersionRange: '>=37 <38',
        gpuVendors: ['nvidia'], gpuDevices: ['nvidia-fixture-device'],
        driverVersions: ['fixture-driver-1'], qualificationSurface: 'test-webgpu',
        driverPolicy: 'exact-receipt-required',
      },
      {
        id: 'macos-arm64-webgpu', os: 'macos', osVersionRange: '>=14 <16',
        architectures: ['arm64'], electronVersionRange: '>=37 <38',
        gpuVendors: ['apple'], gpuDevices: ['apple-fixture-device'],
        driverVersions: ['fixture-driver-1'], qualificationSurface: 'test-webgpu',
        driverPolicy: 'exact-receipt-required',
      },
    ],
  },
  previousRelease: { releaseId: 'fixture-previous', packSemanticRoot: sha('9') },
  rollout: {
    rulesDigest: sha('a'),
    activationAuthority: 'customer',
    selfPromotionAllowed: false,
    stages: [{ id: 'customer-activation', eligibleFleetPercent: 100, requiredObservationDigest: sha('b') }],
  },
  rollback: { releaseId: 'fixture-previous', packSemanticRoot: sha('9'), authority: 'customer' },
  revocation: releaseContract.revocation,
  dataCustody: {
    policyDigest: sha('d'),
    promptRetention: 'none',
    outputRetention: 'none',
    telemetryExport: 'none',
  },
  claimBoundary: { externalCustomer: false, commercialClaimAllowed: false },
};
release.releaseId = `${release.application.applicationId}-${release.candidate.logicalModelId}-release-${hashProductionRelease(release).slice(7, 23)}`;
const manifestPath = path.join(tmpRoot, 'production-release.json');
await fs.writeFile(manifestPath, `${JSON.stringify(release)}\n`, 'utf8');

const applicationReceipt = {
  schema: APPLICATION_GATE_RECEIPT_SCHEMA,
  receiptId: 'application-acceptance',
  releaseId: release.releaseId,
  applicationRevisionDigest: release.application.revisionDigest,
  workload: release.acceptance.workload,
  oracle: release.acceptance.oracle,
  packSemanticRoot: pack.semanticRoot,
  targetPlanId: pack.targetPlans[0].targetId,
  resolvedExecutionId: sha('f'),
  providerId: 'doppler-webgpu',
  deviceTargetId: release.supportedDevices.targets[0].id,
  evaluator: { id: 'fixture-evaluator', revisionDigest: sha('e') },
  status: 'passed',
  observations: {
    quality: 0.95,
    coldLatencyMs: 80,
    warmLatencyMs: 40,
    peakMemoryBytes: 512,
    failureRate: 0,
    startupPassed: true,
    recoveryPassed: true,
  },
  failedSamples: [],
  createdAtUtc: release.createdAtUtc,
  digest: '',
};
applicationReceipt.digest = hashProductionReleaseEvidence(applicationReceipt);

const privatePath = path.join(tmpRoot, 'release.private.json');
const publicPath = path.join(tmpRoot, 'release.public.json');
const packTrustPath = path.join(tmpRoot, 'pack-trust.json');
const fleetTrustPath = path.join(tmpRoot, 'fleet-trust.json');
await Promise.all([
  fs.writeFile(privatePath, JSON.stringify(signingPrivate), 'utf8'),
  fs.writeFile(publicPath, JSON.stringify(signingPublic), 'utf8'),
  fs.writeFile(packTrustPath, JSON.stringify({ [TEST_PACK_AUTHORITY]: TEST_PACK_PUBLIC_KEY }), 'utf8'),
  fs.writeFile(fleetTrustPath, JSON.stringify({ 'fixture-fleet-authority': signingPublic }), 'utf8'),
]);
const common = {
  manifestPath,
  repoRoot: tmpRoot,
  packTrustedSignersPath: packTrustPath,
  signingPrivateKeyPath: privatePath,
  signingPublicKeyPath: publicPath,
  signingAuthority: 'fixture-fleet-authority',
};
const receiptPaths = [];
for (const target of release.supportedDevices.targets) {
  applicationReceipt.deviceTargetId = target.id;
  applicationReceipt.digest = hashProductionReleaseEvidence(applicationReceipt);
  await fs.writeFile(
    path.join(tmpRoot, 'acceptance.js'),
    `process.stdout.write(${JSON.stringify(`${JSON.stringify(applicationReceipt)}\n`)});\n`,
    'utf8'
  );
  const devicePath = path.join(tmpRoot, `${target.id}.device.json`);
  await fs.writeFile(devicePath, JSON.stringify({
    schema: 'doppler.electron-device-identity/v1',
    targetId: target.id,
    os: target.os,
    osVersion: target.os === 'windows' ? '10.0.22631' : '14.5',
    architecture: target.architectures[0],
    electronVersion: '37.1',
    gpuVendor: target.gpuVendors[0],
    gpuDevice: target.gpuDevices[0],
    driverVersion: target.driverVersions[0],
    surface: target.qualificationSurface,
    hasF16: false,
    hasSubgroups: false,
    maxBufferSize: 4,
    observedAtUtc: release.createdAtUtc,
  }), 'utf8');
  const result = await runProductionRelease({
    ...common,
    action: 'qualify',
    outputDirectory: path.join(tmpRoot, 'qualify', target.id),
    targetId: target.id,
    deviceIdentityPath: devicePath,
  });
  assert.equal(result.status, 'passed');
  assert.equal(result.activationPerformed, false);
  assert.equal(JSON.parse(await fs.readFile(result.candidatePackPath, 'utf8')).semanticRoot, pack.semanticRoot);
  receiptPaths.push(result.receiptPath);
}

const rejectedDeviceTarget = release.supportedDevices.targets[0];
const rejectedDevicePath = path.join(tmpRoot, 'rejected-driver.device.json');
await fs.writeFile(rejectedDevicePath, JSON.stringify({
  schema: 'doppler.electron-device-identity/v1',
  targetId: rejectedDeviceTarget.id,
  os: rejectedDeviceTarget.os,
  osVersion: '10.0.22631',
  architecture: rejectedDeviceTarget.architectures[0],
  electronVersion: '37.1',
  gpuVendor: rejectedDeviceTarget.gpuVendors[0],
  gpuDevice: rejectedDeviceTarget.gpuDevices[0],
  driverVersion: 'undeclared-driver',
  surface: rejectedDeviceTarget.qualificationSurface,
  hasF16: false,
  hasSubgroups: false,
  maxBufferSize: 4,
  observedAtUtc: release.createdAtUtc,
}), 'utf8');
await assert.rejects(runProductionRelease({
  ...common,
  action: 'qualify',
  outputDirectory: path.join(tmpRoot, 'rejected-driver'),
  targetId: rejectedDeviceTarget.id,
  deviceIdentityPath: rejectedDevicePath,
}), /driverVersion is outside the target policy/u);

const decisionOutput = path.join(tmpRoot, 'decision');
const decided = await runProductionRelease({
  ...common,
  action: 'decide',
  outputDirectory: decisionOutput,
  fleetReceiptPaths: receiptPaths,
  fleetTrustedSignersPath: fleetTrustPath,
});
assert.equal(decided.eligibility, 'eligible');
assert.equal(decided.activationPerformed, false);
assert.equal(JSON.parse(await fs.readFile(decided.candidatePackPath, 'utf8')).semanticRoot, pack.semanticRoot);
const decision = JSON.parse(await fs.readFile(decided.decisionPath, 'utf8'));
assert.equal(decision.selfPromotionAllowed, false);
assert.equal(decision.activationAuthority, 'customer');
await verifyProductionReleaseEvidenceSignature(decision, {
  'fixture-fleet-authority': signingPublic,
});

const blocked = await runProductionRelease({
  ...common,
  action: 'decide',
  outputDirectory: path.join(tmpRoot, 'blocked'),
  fleetReceiptPaths: receiptPaths.slice(0, 1),
  fleetTrustedSignersPath: fleetTrustPath,
});
assert.equal(blocked.eligibility, 'blocked');
assert.ok(blocked.failureBundlePath);
const failureBundle = JSON.parse(await fs.readFile(blocked.failureBundlePath, 'utf8'));
assert.equal(failureBundle.retained, true);
assert.equal(failureBundle.previousRelease.packSemanticRoot, release.previousRelease.packSemanticRoot);

const extraReceiptValue = JSON.parse(await fs.readFile(receiptPaths[0], 'utf8'));
const extraReceipt = await signProductionReleaseEvidence({
  ...extraReceiptValue,
  receiptId: 'undeclared-target-receipt',
  targetId: 'undeclared-target',
  digest: '',
  signature: null,
}, {
  authority: 'fixture-fleet-authority',
  privateKeyJwk: signingPrivate,
  publicKeyJwk: signingPublic,
});
const extraReceiptPath = path.join(tmpRoot, 'undeclared-target.receipt.json');
await fs.writeFile(extraReceiptPath, JSON.stringify(extraReceipt), 'utf8');
const blockedExtraReceipt = await runProductionRelease({
  ...common,
  action: 'decide',
  outputDirectory: path.join(tmpRoot, 'blocked-extra-receipt'),
  fleetReceiptPaths: [...receiptPaths, extraReceiptPath],
  fleetTrustedSignersPath: fleetTrustPath,
});
assert.equal(blockedExtraReceipt.eligibility, 'blocked');
const blockedExtraDecision = JSON.parse(await fs.readFile(
  blockedExtraReceipt.decisionPath,
  'utf8'
));
assert.ok(blockedExtraDecision.reasons.some((reason) => (
  reason.code === 'unsupported-device' && reason.scope === 'undeclared-target'
)));

console.log('production-release.test: ok');
