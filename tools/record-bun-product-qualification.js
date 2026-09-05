#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { isPlainObject } from '../src/formats/plain-object.js';

import {
  validateBunProductQualificationPolicy,
} from './check-bun-product-qualification.js';
import { computeCanonicalJsonSha256 } from './lib/canonical-json.js';
import {
  BUN_QUALIFICATION_EVIDENCE_CLASSES,
  validateBunProductQualificationEvidence,
} from './lib/bun-product-qualification-evidence.js';
import {
  validateDopplerRuntimeOwnershipReceipt,
} from './lib/runtime-ownership-execution-evidence.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_POLICY_PATH = path.join(REPO_ROOT, 'tools/policies/bun-product-qualification.json');
const CAPTURE_SCHEMA = 'doppler.bun-product-qualification-capture/v1';
const EVIDENCE_FIELDS = Object.freeze([
  'execution',
  ...BUN_QUALIFICATION_EVIDENCE_CLASSES,
]);
const DAY_MS = 24 * 60 * 60 * 1000;

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
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

function hasPriorEvaluation(qualification) {
  return Boolean(
    qualification.resolvedArtifactVariantId
    || qualification.resolvedExecutionId
    || qualification.bunVersion
    || qualification.webgpuImplementationId
    || qualification.providerId
    || qualification.qualifiedAtUtc
    || qualification.expiresAtUtc
    || Object.values(qualification.evidence).some((value) => value !== null)
  );
}

async function validateCapture(capture, context) {
  const { qualification, repoRoot, now, maxAgeDays } = context;
  exactKeys(capture, [
    'schema',
    'qualificationId',
    'qualifiedAtUtc',
    'expiresAtUtc',
    'evidence',
  ], 'Bun qualification capture');
  if (capture.schema !== CAPTURE_SCHEMA) {
    throw new Error(`capture.schema must be ${CAPTURE_SCHEMA}.`);
  }
  if (capture.qualificationId !== qualification.id) {
    throw new Error(`capture.qualificationId must be ${qualification.id}.`);
  }
  const qualifiedAt = instant(capture.qualifiedAtUtc, 'capture.qualifiedAtUtc');
  const expiresAt = instant(capture.expiresAtUtc, 'capture.expiresAtUtc');
  if (qualifiedAt.getTime() > now.getTime()) {
    throw new Error('capture.qualifiedAtUtc must not be in the future.');
  }
  if (expiresAt.getTime() <= qualifiedAt.getTime()) {
    throw new Error('capture.expiresAtUtc must follow capture.qualifiedAtUtc.');
  }
  if (expiresAt.getTime() > qualifiedAt.getTime() + maxAgeDays * DAY_MS) {
    throw new Error(`capture.expiresAtUtc exceeds the ${maxAgeDays}-day policy limit.`);
  }
  exactKeys(capture.evidence, EVIDENCE_FIELDS, 'capture.evidence');
  const evidence = {};
  for (const field of EVIDENCE_FIELDS) {
    evidence[field] = await readEvidence(
      capture.evidence[field],
      `capture.evidence.${field}`,
      repoRoot
    );
  }
  const retainedPaths = Object.values(evidence).map((entry) => entry.path);
  if (new Set(retainedPaths).size !== retainedPaths.length) {
    throw new Error('Bun qualification evidence paths must be distinct.');
  }
  const execution = validateDopplerRuntimeOwnershipReceipt(evidence.execution.receipt, {
    logicalModelId: qualification.logicalModelId,
  });
  if (execution.errors.length > 0) {
    throw new Error(`Bun execution evidence is invalid: ${execution.errors.join('; ')}`);
  }
  if (!execution.resolution) {
    throw new Error('Bun execution evidence must contain resolved artifact and execution identities.');
  }
  if (execution.timestamp?.getTime() > qualifiedAt.getTime()) {
    throw new Error('capture.qualifiedAtUtc must not predate Bun execution evidence.');
  }
  const firstSemantic = evidence[BUN_QUALIFICATION_EVIDENCE_CLASSES[0]].receipt;
  const identity = {
    qualificationId: qualification.id,
    workload: qualification.workload,
    logicalModelId: qualification.logicalModelId,
    manifestVariantId: qualification.manifestVariantId,
    resolvedArtifactVariantId: execution.resolution.resolvedArtifactVariantId,
    resolvedExecutionId: execution.resolution.resolvedExecutionId,
    bunVersion: firstSemantic.bunVersion,
    webgpuImplementationId: firstSemantic.webgpuImplementationId,
    providerId: firstSemantic.providerId,
    correctnessClass: qualification.correctnessClass,
    harnessRevision: firstSemantic.harnessRevision,
    environmentFingerprint: firstSemantic.environmentFingerprint,
  };
  for (const evidenceClass of BUN_QUALIFICATION_EVIDENCE_CLASSES) {
    const result = validateBunProductQualificationEvidence(evidence[evidenceClass].receipt, {
      ...identity,
      evidenceClass,
    });
    if (result.errors.length > 0) {
      throw new Error(`${evidenceClass} evidence is invalid: ${result.errors.join('; ')}`);
    }
    if (result.capturedAt?.getTime() > qualifiedAt.getTime()) {
      throw new Error('capture.qualifiedAtUtc must not predate Bun semantic evidence.');
    }
  }
  return {
    ...identity,
    qualifiedAtUtc: qualifiedAt.toISOString(),
    expiresAtUtc: expiresAt.toISOString(),
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

export async function recordBunProductQualification(options) {
  const repoRoot = path.resolve(options.repoRoot || REPO_ROOT);
  const policyPath = path.resolve(options.policyPath || DEFAULT_POLICY_PATH);
  const capturePath = path.resolve(text(options.capturePath, 'capturePath'));
  const outputPolicyPath = path.resolve(text(options.outputPolicyPath, 'outputPolicyPath'));
  const now = options.now instanceof Date ? options.now : new Date();
  const policy = await readJson(policyPath, 'Bun qualification policy');
  const validationOptions = {
    repoRoot,
    now,
    subsystemsPath: options.subsystemsPath,
    releaseRegistryPath: options.releaseRegistryPath,
    releaseMatrixPath: options.releaseMatrixPath,
    subsystems: options.subsystems,
    releaseRegistry: options.releaseRegistry,
    releaseMatrix: options.releaseMatrix,
  };
  const current = await validateBunProductQualificationPolicy(policy, validationOptions);
  if (current.errors.length > 0) {
    throw new Error(`Bun qualification policy is invalid: ${current.errors[0]}`);
  }
  const capture = await readJson(capturePath, 'Bun qualification capture');
  const qualification = policy.qualifications.find((entry) => entry.id === capture.qualificationId);
  if (!qualification) throw new Error(`Unknown Bun qualification: ${capture.qualificationId}.`);
  if (qualification.claimAllowed) {
    throw new Error(`Recorder cannot replace claimable Bun qualification ${qualification.id}.`);
  }
  if (hasPriorEvaluation(qualification) && options.replace !== true) {
    throw new Error(`Bun qualification ${qualification.id} already has evaluation state; use --replace.`);
  }
  const validated = await validateCapture(capture, {
    qualification,
    repoRoot,
    now,
    maxAgeDays: policy.qualificationMaxAgeDays,
  });
  const outputPolicy = structuredClone(policy);
  const output = outputPolicy.qualifications.find((entry) => entry.id === qualification.id);
  output.resolvedArtifactVariantId = validated.resolvedArtifactVariantId;
  output.resolvedExecutionId = validated.resolvedExecutionId;
  output.bunVersion = validated.bunVersion;
  output.webgpuImplementationId = validated.webgpuImplementationId;
  output.providerId = validated.providerId;
  output.qualifiedAtUtc = validated.qualifiedAtUtc;
  output.expiresAtUtc = validated.expiresAtUtc;
  output.evidence = { ...validated.evidence, promotion: null };
  output.claimAllowed = false;
  output.blockers = ['bun-product-evaluation-awaiting-explicit-promotion'];
  let report = await validateBunProductQualificationPolicy(outputPolicy, validationOptions);
  if (report.errors.length > 0) {
    throw new Error(`Recorded Bun qualification policy is invalid: ${report.errors[0]}`);
  }
  const qualificationReport = report.qualifications.find((entry) => entry.id === qualification.id);
  output.blockers = Array.from(new Set([
    'bun-product-evaluation-awaiting-explicit-promotion',
    ...qualificationReport.reasons,
  ]));
  report = await validateBunProductQualificationPolicy(outputPolicy, validationOptions);
  if (report.errors.length > 0) {
    throw new Error(`Recorded Bun qualification policy is invalid: ${report.errors[0]}`);
  }
  await writeJsonAtomically(outputPolicyPath, outputPolicy);
  return {
    qualificationId: qualification.id,
    outputPolicyPath,
    resolvedArtifactVariantId: output.resolvedArtifactVariantId,
    resolvedExecutionId: output.resolvedExecutionId,
    bunVersion: output.bunVersion,
    webgpuImplementationId: output.webgpuImplementationId,
    providerId: output.providerId,
    claimAllowed: false,
    blockers: output.blockers,
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
  const result = await recordBunProductQualification(options);
  if (options.json) console.log(JSON.stringify(result, null, 2));
  else {
    console.log(
      `bun-product-qualification-record: captured ${result.qualificationId}; `
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
