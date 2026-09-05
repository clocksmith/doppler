#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { isPlainObject } from '../src/formats/plain-object.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_PATHS = Object.freeze({
  contract: path.join(REPO_ROOT, 'tools', 'policies', 'product-portfolio-coherence.json'),
  catalog: path.join(REPO_ROOT, 'models', 'catalog.json'),
  productIntegrations: path.join(
    REPO_ROOT,
    'tools',
    'policies',
    'product-integration-qualification.json'
  ),
  providerConformance: path.join(
    REPO_ROOT,
    'tools',
    'policies',
    'provider-conformance.json'
  ),
  runtimeOwnership: path.join(
    REPO_ROOT,
    'benchmarks',
    'vendors',
    'runtime-ownership-decisions.json'
  ),
  bunQualification: path.join(
    REPO_ROOT,
    'tools',
    'policies',
    'bun-product-qualification.json'
  ),
});
const GOAL_IDS = Object.freeze([
  'local-webgpu-product-surface',
  'model-artifact-runtime-contract',
  'correctness-performance-claims',
]);
const REQUIRED_GATES = Object.freeze([
  'product-integration',
  'provider-conformance',
  'runtime-ownership',
  'bun-product',
]);
const WORKLOADS = Object.freeze(['generation', 'embedding', 'reranking']);
const PRODUCT_WORKLOADS = Object.freeze({
  generation: 'generation',
  embedding: 'embedding-retrieval',
  reranking: 'reranking',
});
const GATE_SOURCES = Object.freeze({
  productIntegration: {
    source: 'productIntegrations',
    rows: 'integrations',
    label: 'product integration',
    manifestVariant: false,
    correctnessClass: false,
  },
  providerConformance: {
    source: 'providerConformance',
    rows: 'suites',
    label: 'provider conformance',
    manifestVariant: true,
    correctnessClass: true,
  },
  runtimeOwnership: {
    source: 'runtimeOwnership',
    rows: 'decisions',
    label: 'runtime ownership',
    manifestVariant: true,
    correctnessClass: true,
  },
  bunQualification: {
    source: 'bunQualification',
    rows: 'qualifications',
    label: 'Bun qualification',
    manifestVariant: true,
    correctnessClass: true,
  },
});
const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function exactKeys(value, fields, label, errors) {
  if (!isPlainObject(value)) {
    errors.push(`${label} must be an object`);
    return false;
  }
  const expected = new Set(fields);
  for (const field of fields) {
    if (!Object.hasOwn(value, field)) errors.push(`${label}.${field} is required`);
  }
  for (const field of Object.keys(value)) {
    if (!expected.has(field)) errors.push(`${label}.${field} is not supported`);
  }
  return true;
}

function exactArray(actual, expected, label, errors) {
  if (!Array.isArray(actual) || JSON.stringify(actual) !== JSON.stringify(expected)) {
    errors.push(`${label} must equal ${JSON.stringify(expected)}`);
  }
}

function requiredText(value, label, errors) {
  const normalized = normalizeText(value);
  if (!normalized) errors.push(`${label} must be a non-empty string`);
  return normalized || null;
}

function rows(value, field, label, errors) {
  const entries = value?.[field];
  if (!Array.isArray(entries)) {
    errors.push(`${label}.${field} must be an array`);
    return [];
  }
  return entries;
}

function indexRows(entries, label, errors) {
  const index = new Map();
  for (const entry of entries) {
    const id = normalizeText(entry?.id);
    if (!id) {
      errors.push(`${label} row id must be a non-empty string`);
      continue;
    }
    if (index.has(id)) errors.push(`${label} contains duplicate row ${id}`);
    index.set(id, entry);
  }
  return index;
}

function compare(actual, expected, label, errors) {
  if (actual !== expected) errors.push(`${label} must be ${String(expected)}; received ${String(actual)}`);
}

function validateResolvedArtifact(value, label, errors) {
  if (value === null) return null;
  const normalized = normalizeText(value).toLowerCase();
  if (!SHA256_PATTERN.test(normalized)) {
    errors.push(`${label} must be a SHA-256 identity or null`);
    return null;
  }
  return normalized;
}

function validateBinding(entry, gate, binding, index, errors) {
  const spec = GATE_SOURCES[gate];
  const label = `portfolio.${entry.workload}.bindings.${gate}`;
  const fields = spec.correctnessClass
    ? ['id', 'workload', 'correctnessClass']
    : ['id', 'workload'];
  if (!exactKeys(binding, fields, label, errors)) return null;
  const id = requiredText(binding.id, `${label}.id`, errors);
  const expectedWorkload = gate === 'productIntegration'
    ? PRODUCT_WORKLOADS[entry.workload]
    : entry.workload;
  compare(binding.workload, expectedWorkload, `${label}.workload`, errors);
  const row = id ? index.get(id) : null;
  if (!row) {
    if (id) errors.push(`${label}.id does not identify a ${spec.label} row: ${id}`);
    return null;
  }
  compare(row.workload, binding.workload, `${spec.label}.${id}.workload`, errors);
  compare(row.logicalModelId, entry.logicalModelId, `${spec.label}.${id}.logicalModelId`, errors);
  if (spec.manifestVariant) {
    compare(
      row.manifestVariantId,
      entry.manifestVariantId,
      `${spec.label}.${id}.manifestVariantId`,
      errors
    );
  }
  if (spec.correctnessClass) {
    compare(
      row.correctnessClass,
      binding.correctnessClass,
      `${spec.label}.${id}.correctnessClass`,
      errors
    );
  }
  return row;
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

export function validateProductPortfolioCoherence(sources) {
  const errors = [];
  const contract = sources.contract;
  exactKeys(
    contract,
    ['$schema', 'schemaVersion', 'source', 'goalIds', 'requiredGates', 'portfolio'],
    'portfolio contract',
    errors
  );
  compare(
    contract?.$schema,
    '../../src/config/schema/product-portfolio-coherence.schema.json',
    'portfolio contract.$schema',
    errors
  );
  compare(contract?.schemaVersion, 1, 'portfolio contract.schemaVersion', errors);
  compare(contract?.source, 'doppler', 'portfolio contract.source', errors);
  exactArray(contract?.goalIds, GOAL_IDS, 'portfolio contract.goalIds', errors);
  exactArray(contract?.requiredGates, REQUIRED_GATES, 'portfolio contract.requiredGates', errors);
  const portfolio = Array.isArray(contract?.portfolio) ? contract.portfolio : [];
  if (!Array.isArray(contract?.portfolio)) errors.push('portfolio contract.portfolio must be an array');
  const indexes = Object.fromEntries(Object.entries(GATE_SOURCES).map(([gate, spec]) => [
    gate,
    indexRows(
      rows(sources[spec.source], spec.rows, spec.label, errors),
      spec.label,
      errors
    ),
  ]));
  const catalogIndex = new Map();
  for (const model of rows(sources.catalog, 'models', 'model catalog', errors)) {
    const modelId = normalizeText(model?.modelId);
    if (!modelId) {
      errors.push('model catalog row modelId must be a non-empty string');
      continue;
    }
    if (catalogIndex.has(modelId)) errors.push(`model catalog contains duplicate model ${modelId}`);
    catalogIndex.set(modelId, model);
  }
  const seenWorkloads = new Set();
  const seenBindings = Object.fromEntries(Object.keys(GATE_SOURCES).map((gate) => [gate, new Set()]));
  const reportEntries = [];
  for (const entry of portfolio) {
    const label = `portfolio.${normalizeText(entry?.workload) || '<missing-workload>'}`;
    if (!exactKeys(entry, ['workload', 'logicalModelId', 'manifestVariantId', 'bindings'], label, errors)) {
      continue;
    }
    const workload = requiredText(entry.workload, `${label}.workload`, errors);
    const logicalModelId = requiredText(entry.logicalModelId, `${label}.logicalModelId`, errors);
    const manifestVariantId = requiredText(
      entry.manifestVariantId,
      `${label}.manifestVariantId`,
      errors
    );
    if (workload && !WORKLOADS.includes(workload)) errors.push(`${label}.workload is not recognized`);
    if (workload && seenWorkloads.has(workload)) errors.push(`portfolio contains duplicate workload ${workload}`);
    if (workload) seenWorkloads.add(workload);
    const catalogModel = logicalModelId ? catalogIndex.get(logicalModelId) : null;
    if (!catalogModel) {
      if (logicalModelId) errors.push(`${label}.logicalModelId is absent from models/catalog.json`);
    } else {
      compare(
        catalogModel.manifestVariantId,
        manifestVariantId,
        `model catalog.${logicalModelId}.manifestVariantId`,
        errors
      );
    }
    const bindings = entry.bindings;
    if (!exactKeys(bindings, Object.keys(GATE_SOURCES), `${label}.bindings`, errors)) continue;
    const gateRows = {};
    for (const gate of Object.keys(GATE_SOURCES)) {
      const bindingId = normalizeText(bindings[gate]?.id);
      if (bindingId && seenBindings[gate].has(bindingId)) {
        errors.push(`${label}.bindings.${gate}.id reuses ${bindingId}`);
      }
      if (bindingId) seenBindings[gate].add(bindingId);
      gateRows[gate] = validateBinding(entry, gate, bindings[gate], indexes[gate], errors);
    }
    const resolvedArtifacts = [];
    for (const [gate, row] of Object.entries(gateRows)) {
      if (!row) continue;
      const identity = validateResolvedArtifact(
        row.resolvedArtifactVariantId,
        `${GATE_SOURCES[gate].label}.${row.id}.resolvedArtifactVariantId`,
        errors
      );
      if (identity) resolvedArtifacts.push([gate, identity]);
    }
    const artifactIds = new Set(resolvedArtifacts.map(([, identity]) => identity));
    if (artifactIds.size > 1) {
      errors.push(
        `${label}: resolved artifact identity differs across gates: `
        + resolvedArtifacts.map(([gate, identity]) => `${gate}=${identity}`).join(', ')
      );
    }
    reportEntries.push({
      workload,
      logicalModelId,
      manifestVariantId,
      resolvedArtifactVariantId: artifactIds.size === 1 ? [...artifactIds][0] : null,
      bindings: Object.fromEntries(Object.entries(bindings).map(([gate, binding]) => [
        gate,
        normalizeText(binding?.id) || null,
      ])),
    });
  }
  exactArray([...seenWorkloads], WORKLOADS, 'portfolio workloads', errors);
  return {
    schema: 'doppler.product-portfolio-coherence-report/v1',
    ok: errors.length === 0,
    workloads: reportEntries,
    requiredGates: [...REQUIRED_GATES],
    errors,
  };
}

export async function buildProductPortfolioCoherenceReport(options = {}) {
  const sources = {};
  for (const [name, defaultPath] of Object.entries(DEFAULT_PATHS)) {
    sources[name] = options[name] ?? await readJson(options[`${name}Path`] ?? defaultPath);
  }
  return validateProductPortfolioCoherence(sources);
}

export async function main(argv = process.argv.slice(2)) {
  const json = argv.includes('--json');
  const unsupported = argv.filter((argument) => argument !== '--json');
  if (unsupported.length > 0) throw new Error(`Unknown argument: ${unsupported[0]}`);
  const report = await buildProductPortfolioCoherenceReport();
  if (json) {
    console.log(JSON.stringify(report, null, 2));
  } else if (report.ok) {
    console.log(
      `product-portfolio-coherence: ok (${report.workloads.length} workloads across `
      + `${report.requiredGates.length} gates)`
    );
  } else {
    for (const error of report.errors) console.error(`product-portfolio-coherence: ${error}`);
  }
  if (!report.ok) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
