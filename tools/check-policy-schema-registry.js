#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_REGISTRY_PATH = path.join(REPO_ROOT, 'src', 'config', 'schema', 'policy-schema-registry.json');
const ID_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function isRepoRelativeJsonPath(value) {
  const normalized = normalizeText(value);
  return Boolean(
    normalized
    && !path.isAbsolute(normalized)
    && !normalized.includes('\\')
    && !normalized.split('/').includes('..')
    && normalized.endsWith('.json')
  );
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

async function assertExistingJson(relativePath, label, errors) {
  if (!isRepoRelativeJsonPath(relativePath)) {
    errors.push(`${label} must be a repo-relative JSON path`);
    return null;
  }
  const absolutePath = path.join(REPO_ROOT, relativePath);
  try {
    await fs.stat(absolutePath);
  } catch {
    errors.push(`${label} does not exist: ${relativePath}`);
    return null;
  }
  return absolutePath;
}

function expectedSchemaRef(policyPath, schemaPath) {
  const fromDir = path.dirname(policyPath);
  return path.relative(fromDir, schemaPath).replaceAll(path.sep, '/');
}

async function validatePolicyEntry(entry, seenIds, errors) {
  const id = normalizeText(entry?.id);
  if (!id) {
    errors.push('policy schema registry entry is missing id');
    return;
  }
  if (!ID_PATTERN.test(id)) {
    errors.push(`${id}: policy schema registry id must be lowercase kebab-case`);
  }
  if (seenIds.has(id)) {
    errors.push(`${id}: duplicate policy schema registry id`);
  }
  seenIds.add(id);
  const policyPath = normalizeText(entry?.policyPath);
  const schemaPath = normalizeText(entry?.schemaPath);
  const [policyAbsolutePath, schemaAbsolutePath] = await Promise.all([
    assertExistingJson(policyPath, `${id}: policyPath`, errors),
    assertExistingJson(schemaPath, `${id}: schemaPath`, errors),
  ]);
  if (!policyAbsolutePath || !schemaAbsolutePath) return;
  const [policy, schema] = await Promise.all([
    readJson(policyAbsolutePath),
    readJson(schemaAbsolutePath),
  ]);
  const expectedRef = expectedSchemaRef(policyPath, schemaPath);
  if (normalizeText(policy.$schema) !== expectedRef) {
    errors.push(`${id}: policy $schema must be ${expectedRef}`);
  }
  if (schema.$schema !== 'https://json-schema.org/draft/2020-12/schema') {
    errors.push(`${id}: schema must use JSON Schema 2020-12`);
  }
  if (!normalizeText(schema.$id)) {
    errors.push(`${id}: schema $id is required`);
  }
  if (schema.additionalProperties !== false) {
    errors.push(`${id}: schema must set additionalProperties false at root`);
  }
}

export async function buildPolicySchemaRegistryReport(options = {}) {
  const registryPath = options.registryPath || DEFAULT_REGISTRY_PATH;
  const registry = await readJson(registryPath);
  const errors = [];
  if (!isPlainObject(registry)) {
    return {
      ok: false,
      registryPath: path.relative(REPO_ROOT, registryPath),
      errors: ['policy schema registry must be an object'],
      policies: 0,
    };
  }
  if (registry.schemaVersion !== 1) {
    errors.push('policy schema registry schemaVersion must be 1');
  }
  if (registry.source !== 'doppler') {
    errors.push('policy schema registry source must be "doppler"');
  }
  const policies = Array.isArray(registry.policies) ? registry.policies : [];
  if (policies.length === 0) {
    errors.push('policy schema registry policies must be a non-empty array');
  }
  const seenIds = new Set();
  for (const entry of policies) {
    await validatePolicyEntry(entry, seenIds, errors);
  }
  return {
    ok: errors.length === 0,
    registryPath: path.relative(REPO_ROOT, registryPath),
    errors,
    policies: policies.length,
  };
}

export async function main(argv = process.argv.slice(2)) {
  const json = argv.includes('--json');
  const unsupported = argv.filter((token) => token !== '--json');
  if (unsupported.length > 0) {
    throw new Error(`Unknown argument: ${unsupported[0]}`);
  }
  const report = await buildPolicySchemaRegistryReport();
  if (json) {
    console.log(JSON.stringify(report, null, 2));
  } else if (report.ok) {
    console.log(`policy-schema-registry: schemas ok (${report.policies} policies)`);
  } else {
    for (const error of report.errors) {
      console.error(`policy-schema-registry: ${error}`);
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
