import { execFile } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { promisify } from 'node:util';
import {
  assertProductionRelease,
  hashProductionRelease,
} from '../config/production-release.js';
import {
  APPLICATION_GATE_RECEIPT_SCHEMA,
  ELECTRON_FLEET_RECEIPT_SCHEMA,
  RELEASE_DECISION_SCHEMA,
  RELEASE_FAILURE_BUNDLE_SCHEMA,
  hashProductionReleaseEvidence,
  signProductionReleaseEvidence,
  validateApplicationGateReceipt,
  validateElectronFleetReceipt,
  validateReleaseDecision,
  verifyProductionReleaseEvidenceSignature,
} from '../config/production-release-evidence.js';
import {
  hashPackV2Envelope,
  verifyPackV2Signature,
} from '../config/pack-v2.js';
import { stableSortObject } from '../formats/stable-sort-object.js';
import { forgeModelPack } from './model-pack-forge.js';
import { loadPackSigningKey, loadPackV2 } from './pack-v2.js';

const execFileAsync = promisify(execFile);
const DEVICE_IDENTITY_SCHEMA = 'doppler.electron-device-identity/v1';

function isObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

async function readJson(filePath, label) {
  const resolved = path.resolve(filePath);
  let value;
  try {
    value = JSON.parse(await fs.readFile(resolved, 'utf8'));
  } catch (error) {
    throw new Error(`${label} at "${filePath}" could not be read: ${error.message}`);
  }
  if (!isObject(value)) throw new Error(`${label} at "${filePath}" must be a JSON object.`);
  return { path: resolved, value };
}

async function writeJsonAtomic(filePath, value) {
  const resolved = path.resolve(filePath);
  const directory = path.dirname(resolved);
  await fs.mkdir(directory, { recursive: true });
  const temporary = `${resolved}.tmp`;
  await fs.writeFile(temporary, `${JSON.stringify(stableSortObject(value), null, 2)}\n`, 'utf8');
  await fs.rename(temporary, resolved);
  return resolved;
}

function resolveWithinRoot(root, relativePath, label) {
  const resolvedRoot = path.resolve(root);
  const resolved = path.resolve(resolvedRoot, relativePath);
  const relation = path.relative(resolvedRoot, resolved);
  if (relation.startsWith('..') || path.isAbsolute(relation)) {
    throw new Error(`${label} must remain within the declared repository root.`);
  }
  return resolved;
}

async function loadSigner(request) {
  if (!request.signingPrivateKeyPath || !request.signingPublicKeyPath || !request.signingAuthority) {
    throw new Error('release requires explicit signingPrivateKeyPath, signingPublicKeyPath, and signingAuthority.');
  }
  return {
    authority: request.signingAuthority,
    privateKeyJwk: await loadPackSigningKey(request.signingPrivateKeyPath),
    publicKeyJwk: await loadPackSigningKey(request.signingPublicKeyPath),
  };
}

async function loadTrustedSigners(filePath, label) {
  const { value } = await readJson(filePath, label);
  for (const [authority, key] of Object.entries(value)) {
    if (!authority || !isObject(key) || key.kty !== 'OKP' || key.crv !== 'Ed25519' || typeof key.x !== 'string') {
      throw new Error(`${label} contains an invalid Ed25519 public JWK for "${authority}".`);
    }
    if ('d' in key) throw new Error(`${label} must not contain private key material.`);
  }
  return value;
}

function compareVersions(left, right) {
  const a = String(left).split('.').map((entry) => Number.parseInt(entry, 10));
  const b = String(right).split('.').map((entry) => Number.parseInt(entry, 10));
  const count = Math.max(a.length, b.length);
  for (let index = 0; index < count; index += 1) {
    const delta = (a[index] || 0) - (b[index] || 0);
    if (delta !== 0) return delta < 0 ? -1 : 1;
  }
  return 0;
}

function satisfiesVersionRange(version, range) {
  const clauses = String(range).trim().split(/\s+/u).filter(Boolean);
  return clauses.every((clause) => {
    const match = /^(>=|<=|>|<|=)?([0-9]+(?:\.[0-9]+)*)$/u.exec(clause);
    if (!match) throw new Error(`Unsupported version range clause "${clause}".`);
    const comparison = compareVersions(version, match[2]);
    const operator = match[1] || '=';
    if (operator === '>=') return comparison >= 0;
    if (operator === '<=') return comparison <= 0;
    if (operator === '>') return comparison > 0;
    if (operator === '<') return comparison < 0;
    return comparison === 0;
  });
}

function validateDeviceIdentity(identity, target) {
  const errors = [];
  if (identity.schema !== DEVICE_IDENTITY_SCHEMA) errors.push(`schema must be ${DEVICE_IDENTITY_SCHEMA}`);
  if (identity.targetId !== target.id) errors.push(`targetId must be ${target.id}`);
  if (identity.os !== target.os) errors.push(`os must be ${target.os}`);
  if (!target.architectures.includes(identity.architecture)) errors.push('architecture is outside the target policy');
  if (!target.gpuVendors.includes(identity.gpuVendor)) errors.push('gpuVendor is outside the target policy');
  for (const field of ['osVersion', 'electronVersion', 'gpuDevice', 'driverVersion', 'observedAtUtc']) {
    if (typeof identity[field] !== 'string' || !identity[field].trim()) errors.push(`${field} is required`);
  }
  if (typeof identity.osVersion === 'string'
    && !satisfiesVersionRange(identity.osVersion, target.osVersionRange)) {
    errors.push(`osVersion ${identity.osVersion} does not satisfy ${target.osVersionRange}`);
  }
  if (typeof identity.electronVersion === 'string'
    && !satisfiesVersionRange(identity.electronVersion, target.electronVersionRange)) {
    errors.push(`electronVersion ${identity.electronVersion} does not satisfy ${target.electronVersionRange}`);
  }
  const observed = new Date(identity.observedAtUtc).getTime();
  if (!Number.isFinite(observed) || new Date(observed).toISOString() !== identity.observedAtUtc) {
    errors.push('observedAtUtc must be an ISO instant');
  }
  if (errors.length > 0) throw new Error(`Electron device identity is outside target policy: ${errors.join('; ')}.`);
  return identity;
}

function bindApplicationGateReceipt(receipt, release) {
  const errors = [];
  if (receipt.releaseId !== release.releaseId) errors.push('releaseId mismatch');
  if (receipt.applicationRevisionDigest !== release.application.revisionDigest) {
    errors.push('applicationRevisionDigest mismatch');
  }
  if (receipt.workload?.id !== release.acceptance.workload.id
    || receipt.workload?.digest !== release.acceptance.workload.digest) errors.push('workload identity mismatch');
  if (receipt.oracle?.id !== release.acceptance.oracle.id
    || receipt.oracle?.digest !== release.acceptance.oracle.digest) errors.push('oracle identity mismatch');
  const threshold = release.acceptance.thresholds;
  const observed = receipt.observations;
  if (observed.quality < threshold.quality.minimum) errors.push('quality threshold failed');
  if (observed.coldLatencyMs > threshold.coldLatencyMs.maximum) errors.push('cold latency threshold failed');
  if (observed.warmLatencyMs > threshold.warmLatencyMs.maximum) errors.push('warm latency threshold failed');
  if (observed.peakMemoryBytes > threshold.peakMemoryBytes.maximum) errors.push('peak memory threshold failed');
  if (observed.failureRate > threshold.failureRate.maximum) errors.push('failure-rate threshold failed');
  if (!observed.startupPassed) errors.push('startup gate failed');
  if (!observed.recoveryPassed) errors.push('recovery gate failed');
  if (receipt.status !== 'passed') errors.push('application gate reported failure');
  return errors;
}

async function executeApplicationGates(release, manifestPath, repoRoot, outputDirectory) {
  const receipts = [];
  const failures = [];
  for (const test of release.acceptance.tests) {
    const cwd = resolveWithinRoot(repoRoot, test.workdir, `acceptance test ${test.id} workdir`);
    try {
      const result = await execFileAsync(test.command[0], test.command.slice(1), {
        cwd,
        timeout: test.timeoutMs,
        maxBuffer: 16 * 1024 * 1024,
        encoding: 'utf8',
        env: {
          ...process.env,
          DOPPLER_PRODUCTION_RELEASE_PATH: manifestPath,
        },
      });
      const receipt = JSON.parse(result.stdout.trim());
      const validation = validateApplicationGateReceipt(receipt);
      if (!validation.ok) throw new Error(validation.errors.join('; '));
      if (receipt.schema !== test.evidenceSchema || receipt.schema !== APPLICATION_GATE_RECEIPT_SCHEMA) {
        throw new Error(`acceptance test emitted unexpected evidence schema "${receipt.schema}"`);
      }
      const bindingErrors = bindApplicationGateReceipt(receipt, release);
      receipts.push(receipt);
      if (bindingErrors.length > 0) failures.push({ testId: test.id, errors: bindingErrors });
      await writeJsonAtomic(path.join(outputDirectory, 'application-gates', `${test.id}.json`), receipt);
    } catch (error) {
      const failure = {
        testId: test.id,
        command: test.command,
        message: error.message,
        stdout: error.stdout || '',
        stderr: error.stderr || '',
      };
      failures.push(failure);
      await writeJsonAtomic(path.join(outputDirectory, 'application-gates', `${test.id}.failure.json`), failure);
    }
  }
  return { receipts, failures };
}

function assertPackReleaseBinding(pack, release) {
  const errors = [];
  if (pack.modelId !== release.candidate.logicalModelId) errors.push('logical model mismatch');
  if (pack.semanticRoot !== release.candidate.packSemanticRoot) errors.push('Pack semantic root mismatch');
  const packRelease = pack.release;
  if (packRelease.source.revision !== release.candidate.sourceRevision) errors.push('source revision mismatch');
  if (packRelease.source.revisionDigest !== release.candidate.sourceRevisionDigest) {
    errors.push('source revision digest mismatch');
  }
  if (packRelease.application.applicationId !== release.application.applicationId) {
    errors.push('application identity mismatch');
  }
  if (packRelease.application.applicationRevision !== release.application.revision
    || packRelease.application.applicationRevisionDigest !== release.application.revisionDigest) {
    errors.push('application revision mismatch');
  }
  for (const field of ['workload', 'oracle']) {
    if (packRelease.application[field].id !== release.acceptance[field].id
      || packRelease.application[field].digest !== release.acceptance[field].digest) {
      errors.push(`${field} identity mismatch`);
    }
  }
  if (packRelease.revocation.authorityId !== release.revocation.authorityId
    || packRelease.revocation.policyDigest !== release.revocation.policyDigest
    || packRelease.revocation.offlineExpirySeconds !== release.revocation.offlineExpirySeconds
    || packRelease.revocation.failClosedAfterExpiry !== release.revocation.failClosedAfterExpiry) {
    errors.push('revocation policy mismatch');
  }
  if (errors.length > 0) throw new Error(`Candidate Pack does not bind production release: ${errors.join('; ')}.`);
}

async function loadBoundPack(release, request, repoRoot) {
  const packPath = resolveWithinRoot(repoRoot, release.candidate.packPath, 'candidate.packPath');
  if (request.forgeConfigPath) {
    const forgeConfig = (await readJson(request.forgeConfigPath, 'release Forge config')).value;
    await forgeModelPack({
      ...forgeConfig,
      repoRoot,
      outputPath: packPath,
      allowDevelopmentSigner: false,
    });
  }
  const pack = await loadPackV2(packPath);
  const trustedPackSigners = await loadTrustedSigners(
    request.packTrustedSignersPath,
    'trusted Pack signers'
  );
  await verifyPackV2Signature(pack, trustedPackSigners);
  assertPackReleaseBinding(pack, release);
  return { pack, packPath };
}

async function qualifyTarget(release, manifestPath, request, repoRoot, outputDirectory) {
  if (!request.targetId || !request.deviceIdentityPath) {
    throw new Error('release qualify requires targetId and deviceIdentityPath.');
  }
  const target = release.supportedDevices.targets.find((entry) => entry.id === request.targetId);
  if (!target) throw new Error(`release target "${request.targetId}" is not declared.`);
  const device = validateDeviceIdentity(
    (await readJson(request.deviceIdentityPath, 'Electron device identity')).value,
    target
  );
  const { pack } = await loadBoundPack(release, request, repoRoot);
  const gates = await executeApplicationGates(release, manifestPath, repoRoot, outputDirectory);
  const applicationGateDigest = gates.receipts.length > 0
    ? hashProductionReleaseEvidence({ receipts: gates.receipts })
    : hashProductionReleaseEvidence({ failures: gates.failures });
  const unsignedReceipt = {
    schema: ELECTRON_FLEET_RECEIPT_SCHEMA,
    receiptId: `${release.releaseId}-${target.id}`.replace(/-release-[0-9a-f]{16}-/u, '-'),
    releaseId: release.releaseId,
    targetId: target.id,
    packSemanticRoot: pack.semanticRoot,
    applicationRevisionDigest: release.application.revisionDigest,
    workload: release.acceptance.workload,
    oracle: release.acceptance.oracle,
    device: {
      os: device.os,
      osVersion: device.osVersion,
      architecture: device.architecture,
      electronVersion: device.electronVersion,
      gpuVendor: device.gpuVendor,
      gpuDevice: device.gpuDevice,
      driverVersion: device.driverVersion,
    },
    applicationGateDigest,
    status: gates.failures.length === 0 ? 'passed' : 'failed',
    createdAtUtc: device.observedAtUtc,
    digest: '',
    signature: null,
  };
  const receipt = await signProductionReleaseEvidence(unsignedReceipt, await loadSigner(request));
  const validation = validateElectronFleetReceipt(receipt);
  if (!validation.ok) throw new Error(`Generated fleet receipt is invalid: ${validation.errors.join('; ')}`);
  const receiptPath = await writeJsonAtomic(
    path.join(outputDirectory, 'qualification-receipts', `${target.id}.json`),
    receipt
  );
  return {
    schema: 'doppler.release-command-result/v1',
    action: 'qualify',
    releaseId: release.releaseId,
    targetId: target.id,
    status: receipt.status,
    receiptPath,
    receiptDigest: receipt.digest,
    failureCount: gates.failures.length,
    activationPerformed: false,
  };
}

async function loadFleetReceipt(filePath, trustedSigners) {
  const receipt = (await readJson(filePath, 'Electron fleet receipt')).value;
  const validation = validateElectronFleetReceipt(receipt);
  if (!validation.ok) throw new Error(validation.errors.join('; '));
  await verifyProductionReleaseEvidenceSignature(receipt, trustedSigners);
  return receipt;
}

function validateFleetBinding(receipt, release, target) {
  const errors = [];
  if (receipt.releaseId !== release.releaseId) errors.push('releaseId mismatch');
  if (receipt.targetId !== target.id) errors.push('targetId mismatch');
  if (receipt.packSemanticRoot !== release.candidate.packSemanticRoot) errors.push('Pack semantic root mismatch');
  if (receipt.applicationRevisionDigest !== release.application.revisionDigest) {
    errors.push('application revision mismatch');
  }
  if (receipt.workload.id !== release.acceptance.workload.id
    || receipt.workload.digest !== release.acceptance.workload.digest) errors.push('workload mismatch');
  if (receipt.oracle.id !== release.acceptance.oracle.id
    || receipt.oracle.digest !== release.acceptance.oracle.digest) errors.push('oracle mismatch');
  if (receipt.device.os !== target.os) errors.push('operating system mismatch');
  if (!target.architectures.includes(receipt.device.architecture)) errors.push('architecture mismatch');
  if (!target.gpuVendors.includes(receipt.device.gpuVendor)) errors.push('GPU vendor mismatch');
  if (!satisfiesVersionRange(receipt.device.osVersion, target.osVersionRange)) errors.push('OS version mismatch');
  if (!satisfiesVersionRange(receipt.device.electronVersion, target.electronVersionRange)) {
    errors.push('Electron version mismatch');
  }
  if (receipt.status !== 'passed') errors.push('fleet qualification failed');
  return errors;
}

async function decideRelease(release, request, repoRoot, outputDirectory) {
  const reasons = [];
  let pack = null;
  let packPath = resolveWithinRoot(repoRoot, release.candidate.packPath, 'candidate.packPath');
  try {
    ({ pack, packPath } = await loadBoundPack(release, request, repoRoot));
  } catch (error) {
    reasons.push({ code: 'artifact-invalid', scope: 'candidate-pack', detail: error.message, evidenceDigests: [] });
  }
  const trustedFleetSigners = await loadTrustedSigners(
    request.fleetTrustedSignersPath,
    'trusted fleet signers'
  );
  const receiptPaths = request.fleetReceiptPaths || [];
  const receipts = [];
  const loadedReceipts = [];
  for (const receiptPath of receiptPaths) {
    try {
      loadedReceipts.push(await loadFleetReceipt(receiptPath, trustedFleetSigners));
    } catch (error) {
      reasons.push({
        code: 'evidence-invalid',
        scope: String(receiptPath),
        detail: error.message,
        evidenceDigests: [],
      });
    }
  }
  for (const target of release.supportedDevices.targets) {
    const matches = loadedReceipts.filter((receipt) => receipt.targetId === target.id);
    if (matches.length !== 1) {
      reasons.push({
        code: 'unsupported-device',
        scope: target.id,
        detail: `Expected exactly one signed receipt for ${target.id}; found ${matches.length}.`,
        evidenceDigests: matches.map((entry) => entry.digest),
      });
      continue;
    }
    const bindingErrors = validateFleetBinding(matches[0], release, target);
    if (bindingErrors.length > 0) {
      reasons.push({
        code: 'application-gate-failed',
        scope: target.id,
        detail: bindingErrors.join('; '),
        evidenceDigests: [matches[0].digest],
      });
    }
    receipts.push(matches[0]);
  }
  const eligibility = reasons.length === 0 ? 'eligible' : 'blocked';
  const unsignedDecision = {
    schema: RELEASE_DECISION_SCHEMA,
    releaseId: release.releaseId,
    productionReleaseDigest: hashProductionRelease(release),
    pack: {
      packId: pack?.packId ?? 'invalid-candidate-pack',
      semanticRoot: pack?.semanticRoot ?? release.candidate.packSemanticRoot,
      envelopeDigest: pack
        ? hashPackV2Envelope(pack)
        : hashProductionReleaseEvidence({ invalidCandidate: release.candidate }),
      path: path.relative(repoRoot, packPath),
    },
    eligibility,
    reasons,
    applicationGateReceipts: receipts.map((entry) => ({
      targetId: entry.targetId,
      digest: entry.applicationGateDigest,
      status: entry.status,
    })),
    fleetReceipts: receipts.map((entry) => ({
      targetId: entry.targetId,
      digest: entry.digest,
      status: entry.status,
    })),
    knownExclusions: pack?.release?.exclusions?.known ?? [],
    previousRelease: release.previousRelease,
    rollback: release.rollback,
    revocation: release.revocation,
    activationAuthority: 'customer',
    selfPromotionAllowed: false,
    createdAtUtc: release.createdAtUtc,
    digest: '',
    signature: null,
  };
  const decision = await signProductionReleaseEvidence(unsignedDecision, await loadSigner(request));
  const validation = validateReleaseDecision(decision);
  if (!validation.ok) throw new Error(`Generated release decision is invalid: ${validation.errors.join('; ')}`);
  const decisionPath = await writeJsonAtomic(path.join(outputDirectory, 'release-decision.json'), decision);
  const exclusionsPath = await writeJsonAtomic(
    path.join(outputDirectory, 'known-exclusions.json'),
    { schema: 'doppler.known-exclusions/v1', releaseId: release.releaseId, exclusions: decision.knownExclusions }
  );
  const rollbackPath = await writeJsonAtomic(path.join(outputDirectory, 'rollback-target.json'), release.rollback);
  const revocationPath = await writeJsonAtomic(path.join(outputDirectory, 'revocation-configuration.json'), release.revocation);
  let failureBundlePath = null;
  if (eligibility === 'blocked') {
    failureBundlePath = await writeJsonAtomic(path.join(outputDirectory, 'failure-bundle.json'), {
      schema: RELEASE_FAILURE_BUNDLE_SCHEMA,
      releaseId: release.releaseId,
      candidatePack: decision.pack,
      previousRelease: release.previousRelease,
      rollback: release.rollback,
      reasons,
      retained: true,
      createdAtUtc: release.createdAtUtc,
    });
  }
  return {
    schema: 'doppler.release-command-result/v1',
    action: 'decide',
    releaseId: release.releaseId,
    eligibility,
    decisionPath,
    decisionDigest: decision.digest,
    exclusionsPath,
    rollbackPath,
    revocationPath,
    failureBundlePath,
    activationPerformed: false,
  };
}

export async function runProductionRelease(request) {
  if (!isObject(request)) throw new Error('release request must be an object.');
  if (request.action !== 'qualify' && request.action !== 'decide') {
    throw new Error('release action must be "qualify" or "decide".');
  }
  if (!request.manifestPath || !request.outputDirectory || !request.packTrustedSignersPath) {
    throw new Error('release requires manifestPath, outputDirectory, and packTrustedSignersPath.');
  }
  const repoRoot = path.resolve(request.repoRoot || process.cwd());
  const manifestFile = await readJson(request.manifestPath, 'production release manifest');
  const release = assertProductionRelease(manifestFile.value);
  const outputDirectory = path.resolve(request.outputDirectory);
  await fs.mkdir(outputDirectory, { recursive: true });
  if (request.action === 'qualify') {
    return qualifyTarget(release, manifestFile.path, request, repoRoot, outputDirectory);
  }
  if (!request.fleetTrustedSignersPath) {
    throw new Error('release decide requires fleetTrustedSignersPath.');
  }
  return decideRelease(release, request, repoRoot, outputDirectory);
}
