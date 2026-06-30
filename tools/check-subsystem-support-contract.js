#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_POLICY_PATH = path.join(REPO_ROOT, 'tools', 'policies', 'subsystem-support-contract.json');
const DEFAULT_PACKAGE_PATH = path.join(REPO_ROOT, 'package.json');
const ID_PATTERN = /^[a-z0-9]+(?:[.-][a-z0-9]+)*$/;
const TIERS = new Set(['tier1', 'experimental', 'internal-only', 'not-applicable']);
const CLAIM_VISIBILITY = new Set(['primary', 'secondary', 'none']);
const SCOPES = new Set([
  'api',
  'benchmark',
  'browser',
  'cli',
  'demo',
  'experimental',
  'format',
  'integration',
  'model',
  'runtime',
  'tooling',
]);

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
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

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

function resolveRepoPath(value, label, errors) {
  if (!isRepoRelativePath(value)) {
    errors.push(`${label} must be a repo-relative path`);
    return null;
  }
  return path.join(REPO_ROOT, normalizeText(value));
}

async function validateExistingPath(relativePath, label, errors) {
  if (!isRepoRelativePath(relativePath)) {
    errors.push(`${label} must be a repo-relative path`);
    return;
  }
  try {
    await fs.stat(path.join(REPO_ROOT, relativePath));
  } catch {
    errors.push(`${label} does not exist: ${relativePath}`);
  }
}

function validateStringArray(value, label, errors) {
  if (!Array.isArray(value) || value.length === 0) {
    errors.push(`${label} must be a non-empty array`);
    return [];
  }
  const seen = new Set();
  const out = [];
  for (const item of value) {
    const normalized = normalizeText(item);
    if (!normalized) {
      errors.push(`${label} entries must be non-empty strings`);
      continue;
    }
    if (seen.has(normalized)) {
      errors.push(`${label} contains duplicate entry ${normalized}`);
      continue;
    }
    seen.add(normalized);
    out.push(normalized);
  }
  return out;
}

function validatePolicy(policy, errors) {
  if (!isPlainObject(policy)) {
    errors.push('subsystem support policy must be an object');
    return false;
  }
  if (policy.schemaVersion !== 1) {
    errors.push('subsystem support policy schemaVersion must be 1');
  }
  if (policy.source !== 'doppler') {
    errors.push('subsystem support policy source must be "doppler"');
  }
  resolveRepoPath(policy.subsystemsPath, 'subsystemsPath', errors);
  return errors.length === 0;
}

function packageExports(packageJson) {
  return isPlainObject(packageJson?.exports) ? packageJson.exports : {};
}

function packageBins(packageJson) {
  return isPlainObject(packageJson?.bin) ? packageJson.bin : {};
}

async function validateSubsystem(subsystem, context) {
  const { packageJson, seenIds, errors } = context;
  const id = normalizeText(subsystem?.id);
  if (!id) {
    errors.push('subsystem is missing id');
    return;
  }
  if (!ID_PATTERN.test(id)) {
    errors.push(`${id}: id must use lowercase dotted/kebab identifier syntax`);
  }
  if (seenIds.has(id)) {
    errors.push(`${id}: duplicate subsystem id`);
  }
  seenIds.add(id);

  for (const field of ['label', 'scope', 'tier', 'owner', 'claimVisibility', 'notes']) {
    if (!normalizeText(subsystem?.[field])) {
      errors.push(`${id}: ${field} is required`);
    }
  }
  if (!SCOPES.has(normalizeText(subsystem?.scope))) {
    errors.push(`${id}: scope is not recognized`);
  }
  if (!TIERS.has(normalizeText(subsystem?.tier))) {
    errors.push(`${id}: tier is not recognized`);
  }
  if (!CLAIM_VISIBILITY.has(normalizeText(subsystem?.claimVisibility))) {
    errors.push(`${id}: claimVisibility is not recognized`);
  }
  for (const field of ['userFacing', 'demoDefault', 'exported']) {
    if (typeof subsystem?.[field] !== 'boolean') {
      errors.push(`${id}: ${field} must be boolean`);
    }
  }

  const docs = validateStringArray(subsystem?.docs, `${id}: docs`, errors);
  const entrypoints = validateStringArray(subsystem?.entrypoints, `${id}: entrypoints`, errors);
  await Promise.all([
    ...docs.map((docPath) => validateExistingPath(docPath, `${id}: docs`, errors)),
    ...entrypoints.map((entrypoint) => validateExistingPath(entrypoint, `${id}: entrypoints`, errors)),
  ]);

  const packageExport = subsystem?.packageExport;
  const command = subsystem?.command;
  if (packageExport !== null && !normalizeText(packageExport)) {
    errors.push(`${id}: packageExport must be a non-empty string or null`);
  }
  if (command !== null && !normalizeText(command)) {
    errors.push(`${id}: command must be a non-empty string or null`);
  }
  if (packageExport && !Object.prototype.hasOwnProperty.call(packageExports(packageJson), packageExport)) {
    errors.push(`${id}: packageExport ${packageExport} is not declared in package.json`);
  }
  if (command && !Object.prototype.hasOwnProperty.call(packageBins(packageJson), command)) {
    errors.push(`${id}: command ${command} is not declared in package.json bin`);
  }
  if (subsystem.exported === true && !packageExport && !command) {
    errors.push(`${id}: exported subsystems must declare packageExport or command`);
  }
  if (subsystem.demoDefault === true && subsystem.userFacing !== true) {
    errors.push(`${id}: demoDefault subsystems must be userFacing`);
  }
  if (subsystem.claimVisibility === 'primary' && subsystem.tier === 'internal-only') {
    errors.push(`${id}: primary claim visibility is not allowed on internal-only subsystems`);
  }
  if (subsystem.claimVisibility === 'primary' && subsystem.userFacing !== true) {
    errors.push(`${id}: primary claim visibility requires userFacing`);
  }
  if (subsystem.tier === 'internal-only' && subsystem.userFacing === true) {
    errors.push(`${id}: internal-only subsystems must not be userFacing`);
  }
}

export async function buildSubsystemSupportContractReport(options = {}) {
  const policyPath = options.policyPath || DEFAULT_POLICY_PATH;
  const errors = [];
  const policy = await readJson(policyPath);
  if (!validatePolicy(policy, errors)) {
    return {
      ok: false,
      policyPath: path.relative(REPO_ROOT, policyPath),
      errors,
      subsystems: 0,
    };
  }
  const subsystemsPath = resolveRepoPath(policy.subsystemsPath, 'subsystemsPath', errors);
  const [subsystemRegistry, packageJson] = await Promise.all([
    readJson(subsystemsPath),
    readJson(options.packagePath || DEFAULT_PACKAGE_PATH),
  ]);
  if (subsystemRegistry.schemaVersion !== 1) {
    errors.push(`${policy.subsystemsPath}: schemaVersion must be 1`);
  }
  if (normalizeText(subsystemRegistry.source) !== 'doppler') {
    errors.push(`${policy.subsystemsPath}: source must be "doppler"`);
  }
  const subsystems = Array.isArray(subsystemRegistry.subsystems) ? subsystemRegistry.subsystems : [];
  if (subsystems.length === 0) {
    errors.push(`${policy.subsystemsPath}: subsystems must be a non-empty array`);
  }
  const seenIds = new Set();
  for (const subsystem of subsystems) {
    await validateSubsystem(subsystem, { packageJson, seenIds, errors });
  }
  return {
    ok: errors.length === 0,
    policyPath: path.relative(REPO_ROOT, policyPath),
    errors,
    subsystems: subsystems.length,
    primaryClaims: subsystems.filter((subsystem) => subsystem?.claimVisibility === 'primary').length,
  };
}

export async function main(argv = process.argv.slice(2)) {
  const json = argv.includes('--json');
  const unsupported = argv.filter((token) => token !== '--json');
  if (unsupported.length > 0) {
    throw new Error(`Unknown argument: ${unsupported[0]}`);
  }
  const report = await buildSubsystemSupportContractReport();
  if (json) {
    console.log(JSON.stringify(report, null, 2));
  } else if (report.ok) {
    console.log(`subsystem-support-contract: support tiers ok (${report.subsystems} subsystems, ${report.primaryClaims} primary)`);
  } else {
    for (const error of report.errors) {
      console.error(`subsystem-support-contract: ${error}`);
    }
  }
  if (!report.ok) {
    process.exitCode = 1;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
