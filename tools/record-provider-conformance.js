#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { validateProviderConformancePolicy } from './check-provider-conformance.js';
import { computeCanonicalJsonSha256 } from './lib/canonical-json.js';
import {
  PROVIDER_CONFORMANCE_EVIDENCE_CLASSES,
  validateProviderConformanceEvidence,
} from './lib/provider-conformance-evidence.js';
import {
  validateDopplerRuntimeOwnershipReceipt,
} from './lib/runtime-ownership-execution-evidence.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_POLICY_PATH = path.join(REPO_ROOT, 'tools', 'policies', 'provider-conformance.json');
const CAPTURE_SCHEMA = 'doppler.provider-conformance-capture/v2';
const CAPTURE_EVIDENCE_FIELDS = Object.freeze([
  ...PROVIDER_CONFORMANCE_EVIDENCE_CLASSES,
  'providerReceipt',
]);
const DAY_MS = 24 * 60 * 60 * 1000;

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function exactKeys(value, fields, label) {
  if (!isPlainObject(value)) throw new Error(`${label} must be an object.`);
  const expected = new Set(fields);
  const unsupported = Object.keys(value).find((field) => !expected.has(field));
  if (unsupported) throw new Error(`${label}.${unsupported} is not supported.`);
  const missing = fields.find((field) => !Object.hasOwn(value, field));
  if (missing) throw new Error(`${label}.${missing} is required.`);
}

function text(value, label) {
  const normalized = normalizeText(value);
  if (!normalized) throw new Error(`${label} must be a non-empty string.`);
  return normalized;
}

function instant(value, label) {
  const normalized = text(value, label);
  const parsed = new Date(normalized);
  if (!Number.isFinite(parsed.getTime()) || parsed.toISOString() !== normalized) {
    throw new Error(`${label} must be an ISO instant.`);
  }
  return parsed;
}

function repoPath(value) {
  const normalized = normalizeText(value);
  return Boolean(
    normalized
    && !path.isAbsolute(normalized)
    && !normalized.includes('\\')
    && !normalized.split('/').includes('..')
  );
}

async function readJson(filePath, label) {
  try {
    return JSON.parse(await fs.readFile(filePath, 'utf8'));
  } catch (error) {
    throw new Error(`${label} is not readable JSON: ${error.message}`);
  }
}

async function readEvidence(value, label, repoRoot) {
  if (!repoPath(value)) throw new Error(`${label} must be a repo-relative path.`);
  const evidencePath = normalizeText(value);
  const receipt = await readJson(path.join(repoRoot, evidencePath), label);
  return {
    path: evidencePath,
    digest: computeCanonicalJsonSha256(receipt),
    receipt,
  };
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
  exactKeys(capture, [
    'schema',
    'suiteId',
    'laneId',
    'qualifiedAtUtc',
    'expiresAtUtc',
    'evidence',
  ], 'provider conformance capture');
  if (capture.schema !== CAPTURE_SCHEMA) {
    throw new Error(`capture.schema must be ${CAPTURE_SCHEMA}.`);
  }
  if (capture.suiteId !== suite.id) throw new Error(`capture.suiteId must be ${suite.id}.`);
  if (capture.laneId !== laneId) throw new Error(`capture.laneId must be ${laneId}.`);
  const qualifiedAt = instant(capture.qualifiedAtUtc, 'capture.qualifiedAtUtc');
  const expiresAt = instant(capture.expiresAtUtc, 'capture.expiresAtUtc');
  if (qualifiedAt.getTime() > now.getTime()) {
    throw new Error('capture.qualifiedAtUtc must not be in the future.');
  }
  if (expiresAt.getTime() <= qualifiedAt.getTime()) {
    throw new Error('capture.expiresAtUtc must follow capture.qualifiedAtUtc.');
  }
  if (expiresAt.getTime() > qualifiedAt.getTime() + qualificationMaxAgeDays * DAY_MS) {
    throw new Error(`capture.expiresAtUtc exceeds the ${qualificationMaxAgeDays}-day policy limit.`);
  }
  exactKeys(capture.evidence, CAPTURE_EVIDENCE_FIELDS, 'capture.evidence');
  const evidence = {};
  for (const field of CAPTURE_EVIDENCE_FIELDS) {
    evidence[field] = await readEvidence(
      capture.evidence[field],
      `capture.evidence.${field}`,
      repoRoot
    );
  }
  const paths = Object.values(evidence).map((entry) => entry.path);
  if (new Set(paths).size !== paths.length) {
    throw new Error('Provider conformance evidence paths must be distinct.');
  }
  const execution = validateDopplerRuntimeOwnershipReceipt(evidence.providerReceipt.receipt, {
    logicalModelId: suite.logicalModelId,
  });
  if (execution.errors.length > 0 || execution.reasons.length > 0) {
    throw new Error(
      `Provider receipt is not a passing local execution: ${[
        ...execution.errors,
        ...execution.reasons,
      ].join('; ')}`
    );
  }
  if (!execution.resolution) {
    throw new Error('Provider receipt must contain resolved artifact and execution identities.');
  }
  if (execution.timestamp?.getTime() > qualifiedAt.getTime()) {
    throw new Error('capture.qualifiedAtUtc must not predate the provider receipt.');
  }
  const firstSemantic = evidence[PROVIDER_CONFORMANCE_EVIDENCE_CLASSES[0]].receipt;
  const identity = {
    suiteId: suite.id,
    laneId,
    workload: suite.workload,
    logicalModelId: suite.logicalModelId,
    manifestVariantId: suite.manifestVariantId,
    resolvedArtifactVariantId: execution.resolution.resolvedArtifactVariantId,
    resolvedExecutionId: execution.resolution.resolvedExecutionId,
    implementationId: firstSemantic.implementationId,
    harnessRevision: firstSemantic.harnessRevision,
    environmentFingerprint: firstSemantic.environmentFingerprint,
    providerReceiptDigest: evidence.providerReceipt.digest,
    declaredOperations: suite.declaredOperations,
    correctnessClass: suite.correctnessClass,
  };
  const results = {};
  const reasons = [];
  const summaries = {};
  for (const evidenceClass of PROVIDER_CONFORMANCE_EVIDENCE_CLASSES) {
    const result = validateProviderConformanceEvidence(evidence[evidenceClass].receipt, {
      ...identity,
      evidenceClass,
    });
    if (result.errors.length > 0) {
      throw new Error(`${evidenceClass} evidence is invalid: ${result.errors.join('; ')}`);
    }
    if (result.capturedAt?.getTime() > qualifiedAt.getTime()) {
      throw new Error('capture.qualifiedAtUtc must not predate semantic provider evidence.');
    }
    results[evidenceClass] = result;
    reasons.push(...result.reasons);
    Object.assign(summaries, result.summary);
  }
  return {
    ...identity,
    qualifiedAtUtc: qualifiedAt.toISOString(),
    expiresAtUtc: expiresAt.toISOString(),
    operations: summaries.operations,
    lifecycle: summaries.lifecycle,
    correctness: summaries.correctness,
    positiveCapture: Object.values(results).every((result) => result.passed === true),
    reasons: Array.from(new Set(reasons)),
    evidence: Object.fromEntries(Object.entries(evidence).map(([field, entry]) => [
      field,
      { path: entry.path, digest: entry.digest },
    ])),
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
  const capturePath = path.resolve(text(options.capturePath, 'capturePath'));
  const outputPolicyPath = path.resolve(text(options.outputPolicyPath, 'outputPolicyPath'));
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
  if (suite.claimAllowed || suite.promotion) {
    throw new Error(`Recorder cannot mutate promoted suite ${suite.id}.`);
  }
  const existingIndex = suite.providers.findIndex((provider) => provider.laneId === laneId);
  const existingProvider = existingIndex >= 0 ? suite.providers[existingIndex] : null;
  if (existingProvider?.claimAllowed || existingProvider?.evidence?.promotion) {
    throw new Error(`Recorder cannot replace promoted provider ${suite.id}/${laneId}.`);
  }
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
    && suite.resolvedArtifactVariantId !== validated.resolvedArtifactVariantId
  ) {
    throw new Error(
      `Capture artifact ${validated.resolvedArtifactVariantId} does not match suite `
      + `${suite.resolvedArtifactVariantId}.`
    );
  }
  const provider = {
    laneId,
    implementationId: validated.implementationId,
    harnessRevision: validated.harnessRevision,
    logicalModelId: suite.logicalModelId,
    manifestVariantId: suite.manifestVariantId,
    resolvedArtifactVariantId: validated.resolvedArtifactVariantId,
    resolvedExecutionId: validated.resolvedExecutionId,
    environmentFingerprint: validated.environmentFingerprint,
    operations: validated.operations,
    lifecycle: validated.lifecycle,
    correctness: validated.correctness,
    qualifiedAtUtc: validated.qualifiedAtUtc,
    expiresAtUtc: validated.expiresAtUtc,
    evidence: { ...validated.evidence, promotion: null },
    claimAllowed: false,
    blockers: Array.from(new Set([
      'provider-capture-awaiting-explicit-promotion',
      'provider-promotion-evidence-missing',
      ...validated.reasons,
    ])),
  };
  const outputPolicy = structuredClone(policy);
  const outputSuite = outputPolicy.suites.find((entry) => entry.id === suite.id);
  if (!outputSuite.resolvedArtifactVariantId) {
    outputSuite.resolvedArtifactVariantId = validated.resolvedArtifactVariantId;
  }
  const outputIndex = outputSuite.providers.findIndex((entry) => entry.laneId === laneId);
  if (outputIndex >= 0) outputSuite.providers[outputIndex] = provider;
  else outputSuite.providers.push(provider);
  outputSuite.promotion = null;
  outputSuite.blockers = updateSuiteBlockers(outputSuite, laneId, true);
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
    positiveCapture: validated.positiveCapture,
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
