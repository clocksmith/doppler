#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { validateProviderConformancePolicy } from './check-provider-conformance.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_POLICY_PATH = path.join(REPO_ROOT, 'tools', 'policies', 'provider-conformance.json');
const CAPTURE_SCHEMA = 'doppler.provider-conformance-capture/v1';
const PROVIDER_RECEIPT_SCHEMA = 'doppler_provider_receipt_v1';
const RESOLUTION_SCHEMA = 'doppler.resolution-identity/v1';
const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;
const LIFECYCLE_STAGES = Object.freeze(['load', 'execute', 'unload']);
const LIFECYCLE_RESULTS = new Set(['passed', 'failed', 'not-run']);
const EVIDENCE_FIELDS = Object.freeze([
  'modelContract',
  'resolutionIdentity',
  'operations',
  'lifecycle',
  'correctness',
  'providerReceipt',
]);

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function assertExactKeys(value, expectedFields, label) {
  if (!isPlainObject(value)) throw new Error(`${label} must be an object.`);
  const expected = new Set(expectedFields);
  const unsupported = Object.keys(value).filter((field) => !expected.has(field));
  if (unsupported.length > 0) {
    throw new Error(`${label} contains unsupported field ${unsupported[0]}.`);
  }
  const missing = expectedFields.filter((field) => !Object.hasOwn(value, field));
  if (missing.length > 0) throw new Error(`${label}.${missing[0]} is required.`);
}

function requireText(value, label) {
  const normalized = normalizeText(value);
  if (!normalized) throw new Error(`${label} must be a non-empty string.`);
  return normalized;
}

function requireSha256(value, label) {
  const normalized = requireText(value, label).toLowerCase();
  if (!SHA256_PATTERN.test(normalized)) throw new Error(`${label} must be a SHA-256 identity.`);
  return normalized;
}

function parseIsoInstant(value, label) {
  const normalized = requireText(value, label);
  const instant = new Date(normalized);
  if (!Number.isFinite(instant.getTime()) || instant.toISOString() !== normalized) {
    throw new Error(`${label} must be an ISO instant.`);
  }
  return instant;
}

function sameMembers(left, right) {
  return [...left].sort().join('\n') === [...right].sort().join('\n');
}

function requireStringArray(value, label) {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`${label} must be a non-empty array.`);
  }
  const values = value.map((entry, index) => requireText(entry, `${label}[${index}]`));
  if (new Set(values).size !== values.length) throw new Error(`${label} must not contain duplicates.`);
  return values;
}

function isRepoRelativePath(value) {
  const normalized = normalizeText(value);
  return Boolean(
    normalized
    && !path.isAbsolute(normalized)
    && !normalized.includes('\\')
    && !normalized.split('/').includes('..')
  );
}

async function requireRepoPath(value, label, repoRoot) {
  if (!isRepoRelativePath(value)) throw new Error(`${label} must be a repo-relative path.`);
  const normalized = normalizeText(value);
  try {
    await fs.stat(path.join(repoRoot, normalized));
  } catch {
    throw new Error(`${label} does not exist: ${normalized}.`);
  }
  return normalized;
}

async function readJson(filePath, label) {
  try {
    return JSON.parse(await fs.readFile(filePath, 'utf8'));
  } catch (error) {
    throw new Error(`${label} is not readable JSON: ${error.message}`);
  }
}

function validateLifecycle(value) {
  assertExactKeys(value, LIFECYCLE_STAGES, 'capture.lifecycle');
  const lifecycle = {};
  for (const stage of LIFECYCLE_STAGES) {
    lifecycle[stage] = requireText(value[stage], `capture.lifecycle.${stage}`);
    if (!LIFECYCLE_RESULTS.has(lifecycle[stage])) {
      throw new Error(`capture.lifecycle.${stage} is not recognized.`);
    }
  }
  return lifecycle;
}

function validateCorrectness(value, suite) {
  assertExactKeys(value, ['class', 'passed'], 'capture.correctness');
  const correctnessClass = requireText(value.class, 'capture.correctness.class');
  if (correctnessClass !== suite.correctnessClass) {
    throw new Error(
      `capture.correctness.class must match suite ${suite.id}: ${suite.correctnessClass}.`
    );
  }
  if (typeof value.passed !== 'boolean') {
    throw new Error('capture.correctness.passed must be boolean.');
  }
  return { class: correctnessClass, passed: value.passed };
}

function validateResolution(receipt, suite, positiveCapture) {
  if (receipt.resolutionStatus === 'unavailable') {
    if (receipt.resolution !== null) {
      throw new Error('Unavailable provider receipt resolution must be null.');
    }
    if (positiveCapture) {
      throw new Error('Passing provider capture requires a resolved provider receipt identity.');
    }
    return null;
  }
  if (receipt.resolutionStatus !== 'resolved' || !isPlainObject(receipt.resolution)) {
    throw new Error('Provider receipt resolutionStatus must be resolved or unavailable.');
  }
  const resolution = receipt.resolution;
  if (resolution.schema !== RESOLUTION_SCHEMA) {
    throw new Error(`Provider receipt resolution must use ${RESOLUTION_SCHEMA}.`);
  }
  const logicalModelId = requireText(
    resolution.logicalModelId,
    'provider receipt resolution.logicalModelId'
  );
  if (logicalModelId !== suite.logicalModelId) {
    throw new Error(
      `Provider receipt logicalModelId ${logicalModelId} does not match suite ${suite.logicalModelId}.`
    );
  }
  return {
    logicalModelId,
    resolvedArtifactVariantId: requireSha256(
      resolution.resolvedArtifactVariantId,
      'provider receipt resolution.resolvedArtifactVariantId'
    ),
    resolvedExecutionId: requireSha256(
      resolution.resolvedExecutionId,
      'provider receipt resolution.resolvedExecutionId'
    ),
  };
}

function captureBlockers({ lifecycle, correctness, resolution, receipt }) {
  const blockers = ['provider-capture-awaiting-explicit-promotion'];
  for (const stage of LIFECYCLE_STAGES) {
    if (lifecycle[stage] !== 'passed') blockers.push(`lifecycle-${stage}-not-passed`);
  }
  if (!correctness.passed) blockers.push('correctness-not-passed');
  if (!resolution) blockers.push('resolution-identity-unavailable');
  if (receipt.failure) blockers.push('provider-receipt-recorded-failure');
  if (receipt.failure?.isSimulated === true) blockers.push('simulated-provider-receipt');
  return blockers;
}

function updateSuiteBlockers(suite, laneId, hasResolution) {
  const obsolete = new Set([
    `${laneId}-qualification-receipt-missing`,
    laneId === 'browser-webgpu' ? 'browser-provider-qualification-receipt-missing' : '',
    laneId === 'node-webgpu' ? 'node-provider-qualification-receipt-missing' : '',
  ]);
  if (hasResolution) obsolete.add('resolved-manifest-sha256-missing');
  const blockers = suite.blockers.filter((blocker) => !obsolete.has(blocker));
  const recordedBlocker = `${laneId}-provider-candidate-recorded-not-promoted`;
  if (!blockers.includes(recordedBlocker)) blockers.push(recordedBlocker);
  return blockers;
}

async function validateCapture(capture, context) {
  const { repoRoot, suite, laneId, qualificationMaxAgeDays, now } = context;
  assertExactKeys(capture, [
    'schema',
    'suiteId',
    'laneId',
    'implementationId',
    'environmentFingerprint',
    'operations',
    'lifecycle',
    'correctness',
    'qualifiedAtUtc',
    'expiresAtUtc',
    'evidence',
  ], 'provider conformance capture');
  if (capture.schema !== CAPTURE_SCHEMA) {
    throw new Error(`capture.schema must be ${CAPTURE_SCHEMA}.`);
  }
  if (capture.suiteId !== suite.id) throw new Error(`capture.suiteId must be ${suite.id}.`);
  if (capture.laneId !== laneId) throw new Error(`capture.laneId must be ${laneId}.`);
  const implementationId = requireText(capture.implementationId, 'capture.implementationId');
  const environmentFingerprint = requireSha256(
    capture.environmentFingerprint,
    'capture.environmentFingerprint'
  );
  const operations = requireStringArray(capture.operations, 'capture.operations');
  if (!sameMembers(operations, suite.declaredOperations)) {
    throw new Error(`capture.operations must match suite ${suite.id}.`);
  }
  const lifecycle = validateLifecycle(capture.lifecycle);
  const correctness = validateCorrectness(capture.correctness, suite);
  const qualifiedAt = parseIsoInstant(capture.qualifiedAtUtc, 'capture.qualifiedAtUtc');
  const expiresAt = parseIsoInstant(capture.expiresAtUtc, 'capture.expiresAtUtc');
  if (qualifiedAt.getTime() > now.getTime()) {
    throw new Error('capture.qualifiedAtUtc must not be in the future.');
  }
  if (expiresAt.getTime() <= qualifiedAt.getTime()) {
    throw new Error('capture.expiresAtUtc must be later than capture.qualifiedAtUtc.');
  }
  const maxExpiry = qualifiedAt.getTime() + qualificationMaxAgeDays * 24 * 60 * 60 * 1000;
  if (expiresAt.getTime() > maxExpiry) {
    throw new Error(`capture.expiresAtUtc exceeds the ${qualificationMaxAgeDays}-day policy limit.`);
  }
  assertExactKeys(capture.evidence, EVIDENCE_FIELDS, 'capture.evidence');
  const evidence = {};
  for (const field of EVIDENCE_FIELDS) {
    evidence[field] = await requireRepoPath(
      capture.evidence[field],
      `capture.evidence.${field}`,
      repoRoot
    );
  }
  const receiptPath = path.join(repoRoot, evidence.providerReceipt);
  const receipt = await readJson(receiptPath, 'provider receipt');
  if (receipt.receiptVersion !== PROVIDER_RECEIPT_SCHEMA) {
    throw new Error(`Provider receipt must use ${PROVIDER_RECEIPT_SCHEMA}.`);
  }
  requireText(receipt.receiptId, 'provider receipt receiptId');
  requireText(receipt.policyMode, 'provider receipt policyMode');
  if (!isPlainObject(receipt.model)) throw new Error('Provider receipt model must be an object.');
  requireText(receipt.model.id, 'provider receipt model.id');
  if (!Number.isFinite(receipt.totalDurationMs) || receipt.totalDurationMs < 0) {
    throw new Error('Provider receipt totalDurationMs must be a non-negative number.');
  }
  if (receipt.source !== 'local') {
    throw new Error('Provider conformance cannot record a fallback provider receipt.');
  }
  if (receipt.fallbackDecision?.executed === true) {
    throw new Error('Provider conformance cannot record a receipt that executed fallback.');
  }
  const receiptTimestamp = parseIsoInstant(receipt.timestamp, 'provider receipt timestamp');
  if (receiptTimestamp.getTime() > qualifiedAt.getTime()) {
    throw new Error('capture.qualifiedAtUtc must not precede the provider receipt timestamp.');
  }
  const positiveCapture = LIFECYCLE_STAGES.every((stage) => lifecycle[stage] === 'passed')
    && correctness.passed;
  if (positiveCapture && receipt.failure !== null) {
    throw new Error('Passing provider capture cannot reference a failed provider receipt.');
  }
  if (positiveCapture && !isPlainObject(receipt.device)) {
    throw new Error('Passing provider capture requires a provider receipt device snapshot.');
  }
  const resolution = validateResolution(receipt, suite, positiveCapture);
  return {
    implementationId,
    environmentFingerprint,
    operations,
    lifecycle,
    correctness,
    qualifiedAtUtc: qualifiedAt.toISOString(),
    expiresAtUtc: expiresAt.toISOString(),
    evidence,
    receipt,
    resolution,
  };
}

async function writeJsonAtomically(filePath, value) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const temporaryPath = `${filePath}.${process.pid}.tmp`;
  try {
    await fs.writeFile(temporaryPath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
    await fs.rename(temporaryPath, filePath);
  } finally {
    await fs.rm(temporaryPath, { force: true });
  }
}

export async function recordProviderConformanceCapture(options) {
  const repoRoot = path.resolve(options.repoRoot || REPO_ROOT);
  const policyPath = path.resolve(options.policyPath || DEFAULT_POLICY_PATH);
  const capturePath = path.resolve(requireText(options.capturePath, 'capturePath'));
  const outputPolicyPath = path.resolve(requireText(options.outputPolicyPath, 'outputPolicyPath'));
  const now = options.now instanceof Date ? options.now : new Date();
  const policy = await readJson(policyPath, 'provider conformance policy');
  const existingReport = await validateProviderConformancePolicy(policy, { repoRoot, now });
  if (existingReport.errors.length > 0) {
    throw new Error(`Provider conformance policy is invalid: ${existingReport.errors[0]}`);
  }
  const capture = await readJson(capturePath, 'provider conformance capture');
  const suite = policy.suites.find((entry) => entry.id === capture.suiteId);
  if (!suite) throw new Error(`Unknown provider conformance suite: ${capture.suiteId}.`);
  const laneId = normalizeText(capture.laneId);
  if (!policy.providerLanes.some((lane) => lane.id === laneId)) {
    throw new Error(`Unknown provider lane: ${laneId}.`);
  }
  if (!suite.requiredProviderLaneIds.includes(laneId)) {
    throw new Error(`Provider lane ${laneId} is not required by suite ${suite.id}.`);
  }
  const existingIndex = suite.providers.findIndex((provider) => provider.laneId === laneId);
  if (existingIndex >= 0 && options.replace !== true) {
    throw new Error(`Suite ${suite.id} already contains provider lane ${laneId}; use --replace.`);
  }
  const validated = await validateCapture(capture, {
    repoRoot,
    suite,
    laneId,
    qualificationMaxAgeDays: policy.qualificationMaxAgeDays,
    now,
  });
  if (
    suite.resolvedArtifactVariantId
    && validated.resolution?.resolvedArtifactVariantId
    && suite.resolvedArtifactVariantId !== validated.resolution.resolvedArtifactVariantId
  ) {
    throw new Error(
      `Capture artifact ${validated.resolution.resolvedArtifactVariantId} does not match suite `
      + `${suite.resolvedArtifactVariantId}.`
    );
  }
  const provider = {
    laneId,
    implementationId: validated.implementationId,
    logicalModelId: suite.logicalModelId,
    manifestVariantId: suite.manifestVariantId,
    resolvedArtifactVariantId: validated.resolution?.resolvedArtifactVariantId ?? null,
    resolvedExecutionId: validated.resolution?.resolvedExecutionId ?? null,
    environmentFingerprint: validated.environmentFingerprint,
    operations: validated.operations,
    lifecycle: validated.lifecycle,
    correctness: validated.correctness,
    qualifiedAtUtc: validated.qualifiedAtUtc,
    expiresAtUtc: validated.expiresAtUtc,
    evidence: validated.evidence,
    claimAllowed: false,
    blockers: captureBlockers(validated),
  };
  const outputPolicy = structuredClone(policy);
  const outputSuite = outputPolicy.suites.find((entry) => entry.id === suite.id);
  if (!outputSuite.resolvedArtifactVariantId && validated.resolution?.resolvedArtifactVariantId) {
    outputSuite.resolvedArtifactVariantId = validated.resolution.resolvedArtifactVariantId;
  }
  const outputIndex = outputSuite.providers.findIndex((entry) => entry.laneId === laneId);
  if (outputIndex >= 0) outputSuite.providers[outputIndex] = provider;
  else outputSuite.providers.push(provider);
  outputSuite.blockers = updateSuiteBlockers(
    outputSuite,
    laneId,
    Boolean(validated.resolution)
  );
  const outputReport = await validateProviderConformancePolicy(outputPolicy, { repoRoot, now });
  if (outputReport.errors.length > 0) {
    throw new Error(`Recorded provider conformance policy is invalid: ${outputReport.errors[0]}`);
  }
  await writeJsonAtomically(outputPolicyPath, outputPolicy);
  return {
    suiteId: suite.id,
    laneId,
    outputPolicyPath,
    resolvedArtifactVariantId: provider.resolvedArtifactVariantId,
    resolvedExecutionId: provider.resolvedExecutionId,
    positiveCapture: validated.lifecycle.load === 'passed'
      && validated.lifecycle.execute === 'passed'
      && validated.lifecycle.unload === 'passed'
      && validated.correctness.passed,
    claimAllowed: false,
    blockers: provider.blockers,
  };
}

export function parseArgs(argv) {
  const options = {
    policyPath: DEFAULT_POLICY_PATH,
    capturePath: '',
    outputPolicyPath: '',
    replace: false,
    apply: false,
    json: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--policy') options.policyPath = argv[++index] || '';
    else if (token === '--capture') options.capturePath = argv[++index] || '';
    else if (token === '--out') options.outputPolicyPath = argv[++index] || '';
    else if (token === '--replace') options.replace = true;
    else if (token === '--apply') options.apply = true;
    else if (token === '--json') options.json = true;
    else throw new Error(`Unknown argument: ${token}`);
  }
  if (!options.capturePath) throw new Error('--capture is required.');
  if (options.apply && options.outputPolicyPath) throw new Error('--apply and --out are mutually exclusive.');
  if (options.apply) options.outputPolicyPath = options.policyPath;
  if (!options.outputPolicyPath) throw new Error('--out is required unless --apply is used.');
  if (!options.apply && path.resolve(options.outputPolicyPath) === path.resolve(options.policyPath)) {
    throw new Error('Writing the source policy requires explicit --apply.');
  }
  return options;
}

export async function main(argv = process.argv.slice(2)) {
  const options = parseArgs(argv);
  const result = await recordProviderConformanceCapture(options);
  if (options.json) console.log(JSON.stringify(result, null, 2));
  else {
    console.log(
      `provider-conformance-record: captured ${result.suiteId}/${result.laneId}; `
      + `claimAllowed=false; output=${path.relative(REPO_ROOT, result.outputPolicyPath)}`
    );
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
