#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import {
  DEFAULT_RESOURCE_TELEMETRY_INTERVAL_MS,
  createResourceTelemetrySampler,
  parseResourceTelemetryMode,
} from './resource-telemetry.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, '..');
const REGISTRY_DIR = path.join(ROOT_DIR, 'benchmarks', 'vendors');
const REGISTRY_PATH = path.join(REGISTRY_DIR, 'registry.json');
const WORKLOADS_PATH = path.join(REGISTRY_DIR, 'workloads.json');
const CAPABILITIES_PATH = path.join(REGISTRY_DIR, 'capabilities.json');
const COMPARE_CONFIG_PATH = path.join(REGISTRY_DIR, 'compare-engines.config.json');
const COMPARE_METRIC_CONTRACT_PATH = path.join(REGISTRY_DIR, 'compare-metrics.json');
const BENCHMARK_POLICY_PATH = path.join(REGISTRY_DIR, 'benchmark-policy.json');
const LOCAL_INFERENCE_CLAIM_MATRIX_PATH = path.join(REGISTRY_DIR, 'local-inference-claim-matrix.json');
const MODEL_CATALOG_PATH = path.join(ROOT_DIR, 'models', 'catalog.json');
const PACKAGE_JSON_PATH = path.join(ROOT_DIR, 'package.json');
const RESULTS_DIR = path.join(REGISTRY_DIR, 'results');
const FIXTURES_DIR = path.join(REGISTRY_DIR, 'fixtures');
const SCHEMA_DIR = path.join(REGISTRY_DIR, 'schema');
const REGISTRY_SCHEMA_PATH = path.join(SCHEMA_DIR, 'registry.schema.json');
const WORKLOADS_SCHEMA_PATH = path.join(SCHEMA_DIR, 'workloads.schema.json');
const CAPABILITIES_SCHEMA_PATH = path.join(SCHEMA_DIR, 'capabilities.schema.json');
const COMPARE_CONFIG_SCHEMA_PATH = path.join(SCHEMA_DIR, 'compare-engines-config.schema.json');
const COMPARE_METRIC_CONTRACT_SCHEMA_PATH = path.join(SCHEMA_DIR, 'metric-contract.schema.json');
const HARNESS_SCHEMA_PATH = path.join(SCHEMA_DIR, 'harness.schema.json');
const RESULT_SCHEMA_PATH = path.join(SCHEMA_DIR, 'result.schema.json');
const RELEASE_MATRIX_SCHEMA_PATH = path.join(SCHEMA_DIR, 'release-matrix.schema.json');
const LOCAL_INFERENCE_CLAIM_MATRIX_SCHEMA_PATH = path.join(SCHEMA_DIR, 'local-inference-claim-matrix.schema.json');
const DEFAULT_RELEASE_MATRIX_OUTPUT_PATH = path.join(REGISTRY_DIR, 'release-matrix.json');
const DEFAULT_RELEASE_MATRIX_MARKDOWN_PATH = path.join(ROOT_DIR, 'docs', 'release-matrix.md');
const DEFAULT_COMMAND_TIMEOUT_MS = 600_000;

function usage() {
  return [
    'Usage:',
    '  node tools/vendor-bench.js list',
    '  node tools/vendor-bench.js validate',
    '  node tools/vendor-bench.js capabilities [--target <id>]',
    '  node tools/vendor-bench.js gap --base <id> --target <id>',
    '  node tools/vendor-bench.js matrix [--compare-result <path>] [--output <path>] [--markdown-output <path>] [--timestamp <iso|ms>] [--include-local-results] [--strict-compare-artifacts]',
    '  node tools/vendor-bench.js show --target <id>',
    '  node tools/vendor-bench.js import --target <id> --input <raw.json> [--output <result.json>] [--timestamp <iso|ms>] [--workload <id>] [--model <id>] [--notes <text>]',
    '  node tools/vendor-bench.js run --target <id> [--timeout-ms <ms>] [--resource-telemetry on|off] [--resource-telemetry-interval-ms <ms>] [--resource-telemetry-samples] [--timestamp <iso|ms>] [--output <result.json>] [--workload <id>] [--model <id>] [--notes <text>] -- <command ...>',
    '  node tools/vendor-bench.js fixtures-restamp [--check] [--fixture <path>]',
    '  --timeout-ms <ms>           Command timeout in milliseconds (default: 600000)',
    `  --resource-telemetry on|off Capture process-tree RAM/CPU, system RAM, and optional ROCm GPU telemetry for run mode (default: off; interval default: ${DEFAULT_RESOURCE_TELEMETRY_INTERVAL_MS}ms)`,
    '  --resource-telemetry-samples Include raw telemetry samples in the result JSON (default: summary only)',
    '  --timestamp <iso|ms>         Override deterministic timestamp for generated record/matrix timestamps',
    '  --include-local-results      Include benchmarks/vendors/results/*.json in matrix discovery (default: fixtures only)',
    '  --strict-compare-artifacts   Fail matrix generation on any auto-discovered compare artifact parse error',
    '  --check                     fixtures-restamp: report compatibility but do not write changes',
    '  --fixture <path>            fixtures-restamp: process a single fixture (default: every committed fixture)',
    '',
    'Notes:',
    '  - `run` expects command stdout to include a JSON object payload.',
    '  - `import` and `run` write normalized records to benchmarks/vendors/results/ by default.',
  ].join('\n');
}

function parseArgs(argv) {
  const booleanFlags = new Set([
    'help',
    'h',
    'include-local-results',
    'strict-compare-artifacts',
    'check',
    'resource-telemetry',
    'resource-telemetry-samples',
  ]);
  const asCommandValue = (value) => {
    if (typeof value !== 'string') return null;
    const normalized = value.trim();
    return normalized === '' ? null : normalized;
  };
  const out = {
    command: asCommandValue(argv[0]),
    flags: {},
    positional: [],
    passthrough: [],
  };

  let inPassthrough = false;
  for (let i = 1; i < argv.length; i += 1) {
    const token = argv[i];
    if (inPassthrough) {
      out.passthrough.push(token);
      continue;
    }
    if (token === '--') {
      inPassthrough = true;
      continue;
    }
    if (!token.startsWith('--')) {
      out.positional.push(token);
      continue;
    }

    const eqIndex = token.indexOf('=');
    const normalizedToken = eqIndex === -1 ? token : token.slice(0, eqIndex);
    const inlineValue = eqIndex === -1 ? null : token.slice(eqIndex + 1);
    const key = normalizedToken.slice(2);

    if (inlineValue !== null) {
      if (inlineValue === '') {
        throw new Error(`Missing value for --${key}`);
      }
      out.flags[key] = inlineValue;
      continue;
    }

    if (booleanFlags.has(key)) {
      const nextValue = argv[i + 1];
      if (nextValue === undefined || nextValue.startsWith('--')) {
        out.flags[key] = 'true';
      } else {
        out.flags[key] = nextValue;
        i += 1;
      }
      continue;
    }

    const value = argv[i + 1];
    if (value === undefined || value.startsWith('--')) {
      throw new Error(`Missing value for --${key}`);
    }
    out.flags[key] = value;
    i += 1;
  }

  return out;
}

function parsePositiveInteger(value, defaultValue, label) {
  if (value == null || value === '') return defaultValue;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || !Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
  return parsed;
}

function parseBooleanFlag(value, defaultValue, label) {
  if (value == null || value === '') return defaultValue;
  const normalized = String(value).trim().toLowerCase();
  if (normalized === 'true' || normalized === '1' || normalized === 'yes' || normalized === 'on') return true;
  if (normalized === 'false' || normalized === '0' || normalized === 'no' || normalized === 'off') return false;
  throw new Error(`${label} must be one of: true, false, 1, 0, yes, no, on, off`);
}

function parseTimestampValue(rawValue, label) {
  if (rawValue == null || rawValue === '') return null;
  if (typeof rawValue !== 'string') {
    throw new Error(`${label} must be a string`);
  }
  const trimmed = rawValue.trim();
  if (trimmed === '') return null;
  const asMs = /^[-+]?\d+$/.test(trimmed) ? Number(trimmed) : NaN;
  const parsed = Number.isFinite(asMs) ? new Date(asMs) : new Date(trimmed);
  if (Number.isNaN(parsed.getTime())) {
    throw new Error(`${label} must be ISO-8601 or epoch milliseconds`);
  }
  return parsed.toISOString();
}

function parseJsonFromStdout(stdout, label) {
  const normalized = String(stdout == null ? '' : stdout);
  if (!normalized.trim()) {
    throw new Error(`Command produced no output for ${label}`);
  }

  const tail = (text, maxChars = 2000) => {
    const str = text == null ? '' : String(text);
    if (str.length <= maxChars) return str;
    return str.slice(str.length - maxChars);
  };

  const asObject = (text) => {
    try {
      const parsed = JSON.parse(text);
      return parsed === null || typeof parsed !== 'object' ? null : parsed;
    } catch {
      return null;
    }
  };

  const direct = asObject(normalized.trim());
  if (direct !== null) return direct;
  throw new Error(`Command output for ${label} was not valid strict JSON object. Output tail:\n${tail(normalized)}`);
}

async function readJson(filePath) {
  const text = await fs.readFile(filePath, 'utf8');
  return JSON.parse(text);
}

async function fileExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

const schemaCache = new Map();

async function loadSchema(schemaPath) {
  if (!schemaCache.has(schemaPath)) {
    schemaCache.set(schemaPath, readJson(schemaPath));
  }
  return schemaCache.get(schemaPath);
}

function isJsonObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function valueType(value) {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'array';
  return typeof value;
}

function ensureSchemaMatch(value, schema, trace) {
  const label = trace || '$';
  if (!schema || !isJsonObject(schema)) {
    throw new Error(`Invalid schema at ${label}: expected object`);
  }

  if (schema.const !== undefined && value !== schema.const) {
    throw new Error(`${label}: value must equal ${JSON.stringify(schema.const)}`);
  }
  if (schema.enum && !schema.enum.includes(value)) {
    throw new Error(`${label}: value must be one of ${schema.enum.map((item) => JSON.stringify(item)).join(', ')}`);
  }

  if (schema.type) {
    const expectedTypes = Array.isArray(schema.type) ? schema.type : [schema.type];
    if (expectedTypes.includes('integer')) {
      if (typeof value !== 'number' || !Number.isInteger(value)) {
        throw new Error(`${label}: expected integer, got ${valueType(value)}`);
      }
    }
    if (!expectedTypes.includes(valueType(value))) {
      if (!(expectedTypes.includes('integer') && typeof value === 'number')) {
        throw new Error(`${label}: expected ${expectedTypes.join(' | ')}, got ${valueType(value)}`);
      }
    }
  }

  if (typeof schema.minLength === 'number' && value !== null && value.length < schema.minLength) {
    throw new Error(`${label}: minimum length is ${schema.minLength}`);
  }
  if (typeof schema.minItems === 'number' && Array.isArray(value) && value.length < schema.minItems) {
    throw new Error(`${label}: minimum items is ${schema.minItems}`);
  }
  if (typeof schema.minProperties === 'number' && isJsonObject(value) && Object.keys(value).length < schema.minProperties) {
    throw new Error(`${label}: minimum properties is ${schema.minProperties}`);
  }
  if (typeof value === 'number') {
    if (typeof schema.minimum === 'number' && value < schema.minimum) {
      throw new Error(`${label}: minimum is ${schema.minimum}`);
    }
    if (typeof schema.exclusiveMinimum === 'number' && value <= schema.exclusiveMinimum) {
      throw new Error(`${label}: exclusive minimum is ${schema.exclusiveMinimum}`);
    }
    if (typeof schema.maximum === 'number' && value > schema.maximum) {
      throw new Error(`${label}: maximum is ${schema.maximum}`);
    }
    if (typeof schema.exclusiveMaximum === 'number' && value >= schema.exclusiveMaximum) {
      throw new Error(`${label}: exclusive maximum is ${schema.exclusiveMaximum}`);
    }
  }

  if (isJsonObject(value) && schema.required) {
    for (const requiredKey of schema.required) {
      if (!Object.prototype.hasOwnProperty.call(value, requiredKey)) {
        throw new Error(`${label}: required property "${requiredKey}" is missing`);
      }
    }
  }

  if (schema.pattern && typeof value === 'string') {
    const expression = new RegExp(schema.pattern);
    if (!expression.test(value)) {
      throw new Error(`${label}: value does not match pattern ${schema.pattern}`);
    }
  }

  if (Array.isArray(value) && schema.items) {
    for (let i = 0; i < value.length; i += 1) {
      ensureSchemaMatch(value[i], schema.items, `${label}[${i}]`);
    }
  }

  if (isJsonObject(value)) {
    if (schema.properties) {
      for (const [key, nestedSchema] of Object.entries(schema.properties)) {
        if (Object.prototype.hasOwnProperty.call(value, key)) {
          ensureSchemaMatch(value[key], nestedSchema, `${label}.${key}`);
        }
      }
    }
    if (schema.additionalProperties === false) {
      const propertyKeys = new Set(schema.properties ? Object.keys(schema.properties) : []);
      for (const key of Object.keys(value)) {
        if (!propertyKeys.has(key)) {
          throw new Error(`${label}: unexpected property "${key}"`);
        }
      }
    }
  }
}

async function assertMatchesSchema(payload, schemaPath, label) {
  const schema = await loadSchema(schemaPath);
  ensureSchemaMatch(payload, schema, label || 'document');
}

function assertRegistryShape(registry) {
  if (!registry || typeof registry !== 'object' || Array.isArray(registry)) {
    throw new Error('registry.json must be an object');
  }
  if (registry.schemaVersion !== 1) {
    throw new Error(`registry schemaVersion must be 1, got ${registry.schemaVersion}`);
  }
  if (!Array.isArray(registry.products) || registry.products.length === 0) {
    throw new Error('registry products must be a non-empty array');
  }
}

function assertWorkloadsShape(workloads) {
  if (!workloads || typeof workloads !== 'object' || Array.isArray(workloads)) {
    throw new Error('workloads.json must be an object');
  }
  if (workloads.schemaVersion !== 1) {
    throw new Error(`workloads schemaVersion must be 1, got ${workloads.schemaVersion}`);
  }
  if (!Array.isArray(workloads.workloads)) {
    throw new Error('workloads.workloads must be an array');
  }
  if (workloads.defaults != null) {
    if (typeof workloads.defaults !== 'object' || Array.isArray(workloads.defaults)) {
      throw new Error('workloads.defaults must be an object when provided');
    }
    const compareEnginesDefault = workloads.defaults.compareEngines;
    if (compareEnginesDefault != null && (typeof compareEnginesDefault !== 'string' || compareEnginesDefault.trim() === '')) {
      throw new Error('workloads.defaults.compareEngines must be a non-empty string when provided');
    }
  }
}

const CAPABILITY_FEATURE_STATUS = Object.freeze({
  supported: 'supported',
  unsupported: 'unsupported',
  unknown: 'unknown',
});
const CAPABILITY_FEATURE_STATUS_TEXT = Object.values(CAPABILITY_FEATURE_STATUS).join(', ');

const CAPABILITY_FEATURE_STATUS_VALUES = new Set(Object.values(CAPABILITY_FEATURE_STATUS));

function normalizeCapabilityFeatureStatus(value) {
  if (value === true) return CAPABILITY_FEATURE_STATUS.supported;
  if (value === false) return CAPABILITY_FEATURE_STATUS.unsupported;
  if (typeof value !== 'string') return null;
  const normalized = value.trim().toLowerCase();
  if (CAPABILITY_FEATURE_STATUS_VALUES.has(normalized)) return normalized;
  return null;
}

function isCapabilityFeatureSupported(value) {
  return normalizeCapabilityFeatureStatus(value) === CAPABILITY_FEATURE_STATUS.supported;
}

function normalizeCapabilityFeatureMap(features, requiredFeatureIds = []) {
  const source = features && typeof features === 'object' && !Array.isArray(features)
    ? features
    : {};
  const out = {};
  for (const [featureId, rawValue] of Object.entries(source)) {
    out[featureId] = normalizeCapabilityFeatureStatus(rawValue) || CAPABILITY_FEATURE_STATUS.unknown;
  }
  for (const requiredFeatureId of requiredFeatureIds) {
    if (typeof requiredFeatureId !== 'string' || requiredFeatureId.trim() === '') continue;
    if (!Object.prototype.hasOwnProperty.call(out, requiredFeatureId)) {
      out[requiredFeatureId] = CAPABILITY_FEATURE_STATUS.unknown;
    }
  }
  return out;
}

function assertCapabilitiesShape(capabilities, knownProductIds = []) {
  if (!capabilities || typeof capabilities !== 'object' || Array.isArray(capabilities)) {
    throw new Error('capabilities.json must be an object');
  }
  if (capabilities.schemaVersion !== 1) {
    throw new Error(`capabilities schemaVersion must be 1, got ${capabilities.schemaVersion}`);
  }
  const benchCatalog = Array.isArray(capabilities.featureCatalog?.bench)
    ? capabilities.featureCatalog.bench
    : null;
  const profileCatalog = Array.isArray(capabilities.featureCatalog?.profile)
    ? capabilities.featureCatalog.profile
    : null;
  if (!benchCatalog || benchCatalog.length === 0) {
    throw new Error('capabilities.featureCatalog.bench must be a non-empty array');
  }
  if (!profileCatalog || profileCatalog.length === 0) {
    throw new Error('capabilities.featureCatalog.profile must be a non-empty array');
  }
  const benchFeatureIds = benchCatalog.map((item) => item?.id).filter((value) => typeof value === 'string' && value.trim() !== '');
  const profileFeatureIds = profileCatalog.map((item) => item?.id).filter((value) => typeof value === 'string' && value.trim() !== '');
  collectUniqueIds(benchFeatureIds, 'bench feature');
  collectUniqueIds(profileFeatureIds, 'profile feature');

  if (!Array.isArray(capabilities.targets) || capabilities.targets.length === 0) {
    throw new Error('capabilities.targets must be a non-empty array');
  }

  const knownSet = new Set(knownProductIds);
  const seen = new Set();
  for (const entry of capabilities.targets) {
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
      throw new Error('capabilities target entries must be objects');
    }
    if (typeof entry.id !== 'string' || entry.id.trim() === '') {
      throw new Error('capabilities target id must be a non-empty string');
    }
    if (knownSet.size > 0 && !knownSet.has(entry.id)) {
      throw new Error(`capabilities target "${entry.id}" has no matching product in registry`);
    }
    if (seen.has(entry.id)) {
      throw new Error(`Duplicate capability target id: ${entry.id}`);
    }
    seen.add(entry.id);

    if (typeof entry.name !== 'string' || entry.name.trim() === '') {
      throw new Error(`capabilities target "${entry.id}" must include a non-empty name`);
    }

    const benchFeatures = entry.bench?.features;
    const profileFeatures = entry.profile?.features;
    if (!benchFeatures || typeof benchFeatures !== 'object' || Array.isArray(benchFeatures)) {
      throw new Error(`capabilities target "${entry.id}" must include bench.features object`);
    }
    if (!profileFeatures || typeof profileFeatures !== 'object' || Array.isArray(profileFeatures)) {
      throw new Error(`capabilities target "${entry.id}" must include profile.features object`);
    }

    for (const key of benchFeatureIds) {
      const value = benchFeatures[key];
      if (!normalizeCapabilityFeatureStatus(value)) {
        throw new Error(
          `capabilities target "${entry.id}" bench.features.${key} `
          + `must be one of: ${CAPABILITY_FEATURE_STATUS_TEXT}`
        );
      }
    }
    for (const key of profileFeatureIds) {
      const value = profileFeatures[key];
      if (!normalizeCapabilityFeatureStatus(value)) {
        throw new Error(
          `capabilities target "${entry.id}" profile.features.${key} `
          + `must be one of: ${CAPABILITY_FEATURE_STATUS_TEXT}`
        );
      }
    }
  }

  for (const productId of knownSet) {
    if (!seen.has(productId)) {
      throw new Error(`capabilities.json missing target entry for product id "${productId}"`);
    }
  }

  for (const targetId of seen) {
    if (!knownSet.has(targetId)) {
      throw new Error(`capabilities has target entry for unknown product id "${targetId}"`);
    }
  }
}

async function assertCapabilitiesEvidencePaths(capabilities) {
  const targets = Array.isArray(capabilities?.targets) ? capabilities.targets : [];
  for (const target of targets) {
    const evidenceRows = Array.isArray(target?.evidence) ? target.evidence : [];
    for (const evidencePath of evidenceRows) {
      if (typeof evidencePath !== 'string' || evidencePath.trim() === '') {
        throw new Error(`capabilities target "${target?.id || 'unknown'}" has an invalid evidence entry`);
      }
      if (/^https?:\/\//i.test(evidencePath)) {
        continue;
      }
      const absolutePath = path.isAbsolute(evidencePath)
        ? evidencePath
        : path.join(ROOT_DIR, evidencePath);
      if (!(await fileExists(absolutePath))) {
        throw new Error(
          `capabilities target "${target?.id || 'unknown'}" references missing evidence path: ${evidencePath}`
        );
      }
    }
  }
}

function assertHarnessShape(harness, expectedId) {
  if (!harness || typeof harness !== 'object' || Array.isArray(harness)) {
    throw new Error('harness must be an object');
  }
  if (harness.schemaVersion !== 1) {
    throw new Error(`harness schemaVersion must be 1, got ${harness.schemaVersion}`);
  }
  if (harness.id !== expectedId) {
    throw new Error(`harness id mismatch: expected ${expectedId}, got ${harness.id}`);
  }
  if (!harness.execution || typeof harness.execution !== 'object') {
    throw new Error('harness.execution must be an object');
  }
  if (harness.execution.mode !== 'external-command' && harness.execution.mode !== 'manual-json') {
    throw new Error(`harness execution mode must be external-command or manual-json, got ${harness.execution.mode}`);
  }
  if (harness.execution.stdoutFormat !== 'json') {
    throw new Error(`harness stdoutFormat must be json, got ${harness.execution.stdoutFormat}`);
  }
  if (!harness.normalization || typeof harness.normalization !== 'object') {
    throw new Error('harness.normalization must be an object');
  }
  if (!harness.normalization.metricPaths || typeof harness.normalization.metricPaths !== 'object') {
    throw new Error('harness.normalization.metricPaths must be an object');
  }
  if (!Array.isArray(harness.normalization.requiredMetrics) || harness.normalization.requiredMetrics.length === 0) {
    throw new Error('harness.normalization.requiredMetrics must be a non-empty array');
  }
  for (const metricName of harness.normalization.requiredMetrics) {
    if (!harness.normalization.metricPaths[metricName]) {
      throw new Error(`required metric "${metricName}" has no path mapping`);
    }
  }
}

function collectUniqueIds(values, label) {
  const seen = new Set();
  for (const value of values) {
    if (seen.has(value)) {
      throw new Error(`Duplicate ${label} id: ${value}`);
    }
    seen.add(value);
  }
}

async function loadRegistryBundle() {
  const registry = await readJson(REGISTRY_PATH);
  const workloads = await readJson(WORKLOADS_PATH);
  await assertMatchesSchema(registry, REGISTRY_SCHEMA_PATH, 'registry.json');
  await assertMatchesSchema(workloads, WORKLOADS_SCHEMA_PATH, 'workloads.json');
  assertRegistryShape(registry);
  assertWorkloadsShape(workloads);
  collectUniqueIds(registry.products.map((item) => item.id), 'product');
  const workloadIds = workloads.workloads.map((item) => item.id);
  collectUniqueIds(workloadIds, 'workload');
  const compareEnginesDefault = workloads?.defaults?.compareEngines;
  if (compareEnginesDefault != null && !workloadIds.includes(compareEnginesDefault)) {
    throw new Error(
      `workloads.defaults.compareEngines must reference an existing workload id, got "${compareEnginesDefault}"`
    );
  }
  return { registry, workloads };
}

async function loadCapabilitiesBundle(registry) {
  const capabilities = await readJson(CAPABILITIES_PATH);
  await assertMatchesSchema(capabilities, CAPABILITIES_SCHEMA_PATH, 'capabilities.json');
  const productIds = Array.isArray(registry?.products) ? registry.products.map((item) => item.id) : [];
  assertCapabilitiesShape(capabilities, productIds);
  await assertCapabilitiesEvidencePaths(capabilities);
  return capabilities;
}

function getByPath(objectValue, dottedPath) {
  const segments = String(dottedPath || '').split('.').filter(Boolean);
  let current = objectValue;
  for (const segment of segments) {
    if (current == null) return undefined;
    if (/^\d+$/.test(segment)) {
      const index = Number(segment);
      if (!Array.isArray(current)) return undefined;
      current = current[index];
      continue;
    }
    if (typeof current !== 'object') return undefined;
    current = current[segment];
  }
  return current;
}

function firstFiniteNumber(objectValue, paths) {
  for (const dottedPath of paths || []) {
    const candidate = getByPath(objectValue, dottedPath);
    if (typeof candidate === 'number' && Number.isFinite(candidate)) {
      return candidate;
    }
  }
  return null;
}

function firstScalar(objectValue, paths) {
  for (const dottedPath of paths || []) {
    const candidate = getByPath(objectValue, dottedPath);
    if (candidate === null || candidate === undefined) continue;
    if (typeof candidate === 'string' || typeof candidate === 'number' || typeof candidate === 'boolean') {
      return candidate;
    }
  }
  return null;
}

const DEFAULT_RESULT_ENVIRONMENT_PATHS = Object.freeze({
  hostPlatform: Object.freeze([
    'environment.host.platform',
    'host.platform',
    'env.hostPlatform',
    'result.environment.host.platform',
    'result.host.platform',
    'result.env.hostPlatform',
  ]),
  hostArch: Object.freeze([
    'environment.host.arch',
    'host.arch',
    'env.hostArch',
    'result.environment.host.arch',
    'result.host.arch',
    'result.env.hostArch',
  ]),
  hostNodeVersion: Object.freeze([
    'environment.host.nodeVersion',
    'host.nodeVersion',
    'env.nodeVersion',
    'result.environment.host.nodeVersion',
    'result.host.nodeVersion',
    'result.env.nodeVersion',
  ]),
  hostOsRelease: Object.freeze([
    'environment.host.osRelease',
    'host.osRelease',
    'env.osRelease',
    'result.environment.host.osRelease',
    'result.host.osRelease',
    'result.env.osRelease',
  ]),
  hostCpuModel: Object.freeze([
    'environment.host.cpuModel',
    'host.cpuModel',
    'env.cpuModel',
    'result.environment.host.cpuModel',
    'result.host.cpuModel',
    'result.env.cpuModel',
  ]),
  browserUserAgent: Object.freeze([
    'environment.browser.userAgent',
    'browser.userAgent',
    'env.browser.userAgent',
    'env.browserUserAgent',
    'result.environment.browser.userAgent',
    'result.browser.userAgent',
    'result.env.browser.userAgent',
    'result.env.browserUserAgent',
    'result.result.env.browserUserAgent',
  ]),
  browserPlatform: Object.freeze([
    'environment.browser.platform',
    'browser.platform',
    'env.browser.platform',
    'env.browserPlatform',
    'result.environment.browser.platform',
    'result.browser.platform',
    'result.env.browser.platform',
    'result.env.browserPlatform',
    'result.result.env.browserPlatform',
  ]),
  browserLanguage: Object.freeze([
    'environment.browser.language',
    'browser.language',
    'env.browser.language',
    'env.browserLanguage',
    'result.environment.browser.language',
    'result.browser.language',
    'result.env.browser.language',
    'result.env.browserLanguage',
    'result.result.env.browserLanguage',
  ]),
  browserVendor: Object.freeze([
    'environment.browser.vendor',
    'browser.vendor',
    'env.browser.vendor',
    'env.browserVendor',
    'result.environment.browser.vendor',
    'result.browser.vendor',
    'result.env.browser.vendor',
    'result.env.browserVendor',
    'result.result.env.browserVendor',
  ]),
  browserExecutable: Object.freeze([
    'environment.browser.executable',
    'browser.executable',
    'env.browserExecutable',
    'result.environment.browser.executable',
    'result.browser.executable',
    'result.env.browserExecutable',
  ]),
  browserChannel: Object.freeze([
    'environment.browser.channel',
    'browser.channel',
    'env.browserChannel',
    'result.environment.browser.channel',
    'result.browser.channel',
    'result.env.browserChannel',
  ]),
  gpuApi: Object.freeze([
    'environment.gpu.api',
    'gpu.api',
    'env.gpuApi',
    'result.environment.gpu.api',
    'result.gpu.api',
    'result.env.gpuApi',
  ]),
  gpuBackend: Object.freeze([
    'environment.gpu.backend',
    'gpu.backend',
    'env.webgpuBackend',
    'env.gpuBackend',
    'env.graphicsBackend',
    'result.environment.gpu.backend',
    'result.gpu.backend',
    'result.env.webgpuBackend',
    'result.env.gpuBackend',
    'result.env.graphicsBackend',
  ]),
  gpuVendor: Object.freeze([
    'environment.gpu.vendor',
    'gpu.vendor',
    'deviceInfo.adapterInfo.vendor',
    'deviceInfo.vendor',
    'result.environment.gpu.vendor',
    'result.gpu.vendor',
    'result.deviceInfo.adapterInfo.vendor',
    'result.deviceInfo.vendor',
    'result.result.deviceInfo.adapterInfo.vendor',
    'result.result.deviceInfo.vendor',
  ]),
  gpuArchitecture: Object.freeze([
    'environment.gpu.architecture',
    'gpu.architecture',
    'deviceInfo.adapterInfo.architecture',
    'deviceInfo.architecture',
    'result.environment.gpu.architecture',
    'result.gpu.architecture',
    'result.deviceInfo.adapterInfo.architecture',
    'result.deviceInfo.architecture',
    'result.result.deviceInfo.adapterInfo.architecture',
    'result.result.deviceInfo.architecture',
  ]),
  gpuDevice: Object.freeze([
    'environment.gpu.device',
    'gpu.device',
    'deviceInfo.adapterInfo.device',
    'deviceInfo.device',
    'result.environment.gpu.device',
    'result.gpu.device',
    'result.deviceInfo.adapterInfo.device',
    'result.deviceInfo.device',
    'result.result.deviceInfo.adapterInfo.device',
    'result.result.deviceInfo.device',
  ]),
  gpuDescription: Object.freeze([
    'environment.gpu.description',
    'gpu.description',
    'deviceInfo.adapterInfo.description',
    'deviceInfo.description',
    'result.environment.gpu.description',
    'result.gpu.description',
    'result.deviceInfo.adapterInfo.description',
    'result.deviceInfo.description',
    'result.result.deviceInfo.adapterInfo.description',
    'result.result.deviceInfo.description',
  ]),
  gpuHasF16: Object.freeze([
    'environment.gpu.hasF16',
    'gpu.hasF16',
    'deviceInfo.hasF16',
    'result.environment.gpu.hasF16',
    'result.gpu.hasF16',
    'result.deviceInfo.hasF16',
    'result.result.deviceInfo.hasF16',
  ]),
  gpuHasSubgroups: Object.freeze([
    'environment.gpu.hasSubgroups',
    'gpu.hasSubgroups',
    'deviceInfo.hasSubgroups',
    'result.environment.gpu.hasSubgroups',
    'result.gpu.hasSubgroups',
    'result.deviceInfo.hasSubgroups',
    'result.result.deviceInfo.hasSubgroups',
  ]),
  gpuHasTimestampQuery: Object.freeze([
    'environment.gpu.hasTimestampQuery',
    'gpu.hasTimestampQuery',
    'deviceInfo.hasTimestampQuery',
    'result.environment.gpu.hasTimestampQuery',
    'result.gpu.hasTimestampQuery',
    'result.deviceInfo.hasTimestampQuery',
    'result.result.deviceInfo.hasTimestampQuery',
  ]),
  runtimeLibrary: Object.freeze([
    'environment.runtime.library',
    'runtime.library',
    'env.library',
    'result.environment.runtime.library',
    'result.runtime.library',
    'result.env.library',
    'result.result.env.library',
  ]),
  runtimeVersion: Object.freeze([
    'environment.runtime.version',
    'runtime.version',
    'env.version',
    'result.environment.runtime.version',
    'result.runtime.version',
    'result.env.version',
    'result.result.env.version',
  ]),
  runtimeSurface: Object.freeze([
    'environment.runtime.surface',
    'runtime.surface',
    'surface',
    'env.runtime',
    'result.environment.runtime.surface',
    'result.runtime.surface',
    'result.surface',
    'result.env.runtime',
    'result.result.env.runtime',
  ]),
  runtimeDevice: Object.freeze([
    'environment.runtime.device',
    'runtime.device',
    'env.device',
    'result.environment.runtime.device',
    'result.runtime.device',
    'result.env.device',
    'result.result.env.device',
  ]),
  runtimeDtype: Object.freeze([
    'environment.runtime.dtype',
    'runtime.dtype',
    'env.dtype',
    'result.environment.runtime.dtype',
    'result.runtime.dtype',
    'result.env.dtype',
    'result.result.env.dtype',
  ]),
  runtimeRequestedDtype: Object.freeze([
    'environment.runtime.requestedDtype',
    'runtime.requestedDtype',
    'env.requestedDtype',
    'result.environment.runtime.requestedDtype',
    'result.runtime.requestedDtype',
    'result.env.requestedDtype',
    'result.result.env.requestedDtype',
  ]),
  runtimeExecutionProviderMode: Object.freeze([
    'environment.runtime.executionProviderMode',
    'runtime.executionProviderMode',
    'env.executionProviderMode',
    'result.environment.runtime.executionProviderMode',
    'result.runtime.executionProviderMode',
    'result.env.executionProviderMode',
    'result.result.env.executionProviderMode',
  ]),
  runtimeCacheMode: Object.freeze([
    'environment.runtime.cacheMode',
    'runtime.cacheMode',
    'timing.cacheMode',
    'cacheMode',
    'result.environment.runtime.cacheMode',
    'result.runtime.cacheMode',
    'result.timing.cacheMode',
    'result.cacheMode',
    'result.result.timing.cacheMode',
  ]),
  runtimeLoadMode: Object.freeze([
    'environment.runtime.loadMode',
    'runtime.loadMode',
    'timing.loadMode',
    'loadMode',
    'result.environment.runtime.loadMode',
    'result.runtime.loadMode',
    'result.timing.loadMode',
    'result.loadMode',
    'result.result.timing.loadMode',
  ]),
});

function asNonEmptyStringValue(value) {
  if (value == null) return null;
  const normalized = String(value).trim();
  return normalized === '' ? null : normalized;
}

function firstStringByPathList(objectValue, paths) {
  const value = firstScalar(objectValue, paths);
  return asNonEmptyStringValue(value);
}

function firstBooleanByPathList(objectValue, paths) {
  for (const dottedPath of paths || []) {
    const candidate = getByPath(objectValue, dottedPath);
    if (typeof candidate === 'boolean') {
      return candidate;
    }
    if (typeof candidate === 'string') {
      const normalized = candidate.trim().toLowerCase();
      if (normalized === 'true') return true;
      if (normalized === 'false') return false;
    }
  }
  return null;
}

function readCommandFlagValue(commandParts, flagName) {
  if (!Array.isArray(commandParts) || !flagName) return null;
  for (let i = 0; i < commandParts.length; i += 1) {
    const token = String(commandParts[i] ?? '');
    if (token === flagName) {
      const value = asNonEmptyStringValue(commandParts[i + 1]);
      return value;
    }
    if (token.startsWith(`${flagName}=`)) {
      return asNonEmptyStringValue(token.slice(flagName.length + 1));
    }
  }
  return null;
}

function normalizeGpuBackendLabel(value) {
  const raw = asNonEmptyStringValue(value);
  if (!raw) return null;
  const normalized = raw.toLowerCase();
  if (normalized.includes('metal')) return 'metal';
  if (normalized.includes('vulkan')) return 'vulkan';
  if (normalized.includes('d3d12')) return 'd3d12';
  if (normalized.includes('d3d11')) return 'd3d11';
  if (normalized.includes('opengl') || normalized === 'gl') return 'opengl';
  if (normalized.includes('swiftshader')) return 'swiftshader';
  return normalized;
}

function inferGpuBackendFromCommand(commandParts, hostPlatform) {
  const angleValue = readCommandFlagValue(commandParts, '--use-angle');
  const explicit = normalizeGpuBackendLabel(angleValue);
  if (explicit) return explicit;

  const tokens = Array.isArray(commandParts)
    ? commandParts.map((value) => String(value ?? '').toLowerCase())
    : [];
  if (tokens.some((token) => token.includes('vulkan'))) return 'vulkan';
  if (tokens.some((token) => token.includes('d3d12'))) return 'd3d12';
  if (tokens.some((token) => token.includes('d3d11'))) return 'd3d11';
  if (tokens.some((token) => token.includes('metal'))) return 'metal';

  const platform = asNonEmptyStringValue(hostPlatform);
  if (platform === 'darwin') return 'metal';
  if (platform === 'linux') return 'vulkan';
  if (platform === 'win32') return 'd3d12';
  return null;
}

function resolveEnvironmentPathOverrides(harness) {
  const overrides = harness?.normalization?.environmentPaths;
  if (!overrides || typeof overrides !== 'object' || Array.isArray(overrides)) {
    return {};
  }
  return overrides;
}

function resolveEnvironmentString(rawResult, pathOverrides, key) {
  const overridePaths = Array.isArray(pathOverrides[key]) ? pathOverrides[key] : [];
  const fallbackPaths = DEFAULT_RESULT_ENVIRONMENT_PATHS[key] || [];
  const paths = overridePaths.length > 0 ? overridePaths : fallbackPaths;
  return firstStringByPathList(rawResult, paths);
}

function resolveEnvironmentBoolean(rawResult, pathOverrides, key) {
  const overridePaths = Array.isArray(pathOverrides[key]) ? pathOverrides[key] : [];
  const fallbackPaths = DEFAULT_RESULT_ENVIRONMENT_PATHS[key] || [];
  const paths = overridePaths.length > 0 ? overridePaths : fallbackPaths;
  return firstBooleanByPathList(rawResult, paths);
}

function buildNormalizedEnvironment(rawResult, harness, source) {
  const pathOverrides = resolveEnvironmentPathOverrides(harness);
  const sourceHost = source?.host && typeof source.host === 'object' ? source.host : {};
  const sourceCommand = Array.isArray(source?.command) ? source.command : [];
  const hostPlatform = resolveEnvironmentString(rawResult, pathOverrides, 'hostPlatform')
    || asNonEmptyStringValue(sourceHost.platform);
  const hostArch = resolveEnvironmentString(rawResult, pathOverrides, 'hostArch')
    || asNonEmptyStringValue(sourceHost.arch);
  const hostNodeVersion = resolveEnvironmentString(rawResult, pathOverrides, 'hostNodeVersion')
    || asNonEmptyStringValue(sourceHost.nodeVersion);
  const hostOsRelease = resolveEnvironmentString(rawResult, pathOverrides, 'hostOsRelease')
    || asNonEmptyStringValue(sourceHost.osRelease);
  const hostCpuModel = resolveEnvironmentString(rawResult, pathOverrides, 'hostCpuModel')
    || asNonEmptyStringValue(sourceHost.cpuModel);

  const browserExecutable = resolveEnvironmentString(rawResult, pathOverrides, 'browserExecutable')
    || readCommandFlagValue(sourceCommand, '--browser-executable');
  const browserChannel = resolveEnvironmentString(rawResult, pathOverrides, 'browserChannel')
    || readCommandFlagValue(sourceCommand, '--browser-channel');

  const runtimeDevice = resolveEnvironmentString(rawResult, pathOverrides, 'runtimeDevice');
  const runtimeLibrary = resolveEnvironmentString(rawResult, pathOverrides, 'runtimeLibrary');
  const runtimeSurface = resolveEnvironmentString(rawResult, pathOverrides, 'runtimeSurface');
  const runtimeLoadMode = resolveEnvironmentString(rawResult, pathOverrides, 'runtimeLoadMode');
  const runtimeCacheMode = resolveEnvironmentString(rawResult, pathOverrides, 'runtimeCacheMode');

  const gpuApiRaw = resolveEnvironmentString(rawResult, pathOverrides, 'gpuApi');
  const gpuApi = normalizeGpuBackendLabel(gpuApiRaw)
    || (typeof runtimeDevice === 'string' && runtimeDevice.toLowerCase().includes('webgpu') ? 'webgpu' : null);
  const gpuBackend = normalizeGpuBackendLabel(resolveEnvironmentString(rawResult, pathOverrides, 'gpuBackend'))
    || inferGpuBackendFromCommand(sourceCommand, hostPlatform);

  return {
    host: {
      platform: hostPlatform,
      arch: hostArch,
      nodeVersion: hostNodeVersion,
      osRelease: hostOsRelease,
      cpuModel: hostCpuModel,
    },
    browser: {
      userAgent: resolveEnvironmentString(rawResult, pathOverrides, 'browserUserAgent'),
      platform: resolveEnvironmentString(rawResult, pathOverrides, 'browserPlatform'),
      language: resolveEnvironmentString(rawResult, pathOverrides, 'browserLanguage'),
      vendor: resolveEnvironmentString(rawResult, pathOverrides, 'browserVendor'),
      executable: browserExecutable,
      channel: browserChannel,
    },
    gpu: {
      api: gpuApi,
      backend: gpuBackend,
      vendor: resolveEnvironmentString(rawResult, pathOverrides, 'gpuVendor'),
      architecture: resolveEnvironmentString(rawResult, pathOverrides, 'gpuArchitecture'),
      device: resolveEnvironmentString(rawResult, pathOverrides, 'gpuDevice'),
      description: resolveEnvironmentString(rawResult, pathOverrides, 'gpuDescription'),
      hasF16: resolveEnvironmentBoolean(rawResult, pathOverrides, 'gpuHasF16'),
      hasSubgroups: resolveEnvironmentBoolean(rawResult, pathOverrides, 'gpuHasSubgroups'),
      hasTimestampQuery: resolveEnvironmentBoolean(rawResult, pathOverrides, 'gpuHasTimestampQuery'),
    },
    runtime: {
      library: runtimeLibrary,
      version: resolveEnvironmentString(rawResult, pathOverrides, 'runtimeVersion'),
      surface: runtimeSurface,
      device: runtimeDevice,
      dtype: resolveEnvironmentString(rawResult, pathOverrides, 'runtimeDtype'),
      requestedDtype: resolveEnvironmentString(rawResult, pathOverrides, 'runtimeRequestedDtype'),
      executionProviderMode: resolveEnvironmentString(rawResult, pathOverrides, 'runtimeExecutionProviderMode'),
      cacheMode: runtimeCacheMode,
      loadMode: runtimeLoadMode,
    },
  };
}

function assertRunEnvironmentCompleteness(environment, options = {}) {
  const requireGpu = options.requireGpu === true;
  const missing = [];
  if (!asNonEmptyStringValue(environment?.host?.platform)) missing.push('environment.host.platform');
  if (!asNonEmptyStringValue(environment?.host?.arch)) missing.push('environment.host.arch');
  if (!asNonEmptyStringValue(environment?.host?.nodeVersion)) missing.push('environment.host.nodeVersion');
  if (!asNonEmptyStringValue(environment?.runtime?.library)) missing.push('environment.runtime.library');
  if (!asNonEmptyStringValue(environment?.runtime?.device)) missing.push('environment.runtime.device');
  const browserRequired = asNonEmptyStringValue(environment?.runtime?.surface) === 'browser' || requireGpu;
  if (browserRequired && !asNonEmptyStringValue(environment?.browser?.userAgent)) {
    missing.push('environment.browser.userAgent');
  }
  if (browserRequired && !asNonEmptyStringValue(environment?.browser?.platform)) {
    missing.push('environment.browser.platform');
  }
  if (requireGpu) {
    if (!asNonEmptyStringValue(environment?.gpu?.backend)) missing.push('environment.gpu.backend');
    const gpuIdentity = [
      asNonEmptyStringValue(environment?.gpu?.vendor),
      asNonEmptyStringValue(environment?.gpu?.device),
      asNonEmptyStringValue(environment?.gpu?.description),
    ].filter(Boolean);
    if (gpuIdentity.length === 0) {
      missing.push('environment.gpu.[vendor|device|description]');
    }
  }
  if (missing.length > 0) {
    throw new Error(`missing required runtime environment capture fields: ${missing.join(', ')}`);
  }
}

function assertResultRecordShape(record) {
  if (!record || typeof record !== 'object' || Array.isArray(record)) {
    throw new Error('Normalized record must be an object');
  }

  const allowedTopLevelKeys = new Set([
    'schemaVersion',
    'timestamp',
    'target',
    'harness',
    'workload',
    'model',
    'metrics',
    'source',
    'environment',
    'notes',
    'metadata',
    'resourceTelemetry',
  ]);
  for (const key of Object.keys(record)) {
    if (!allowedTopLevelKeys.has(key)) {
      throw new Error(`record contains unsupported top-level field: ${key}`);
    }
  }

  if (record.schemaVersion !== 1) {
    throw new Error(`record.schemaVersion must be 1, got ${record.schemaVersion}`);
  }
  if (typeof record.timestamp !== 'string' || record.timestamp.trim() === '') {
    throw new Error('record.timestamp is required and must be a non-empty string');
  }
  if (!record.target || typeof record.target !== 'object' || typeof record.target.id !== 'string' || typeof record.target.name !== 'string') {
    throw new Error('record.target.id and record.target.name are required strings');
  }
  if (!record.harness || typeof record.harness !== 'object' || typeof record.harness.id !== 'string' || typeof record.harness.name !== 'string') {
    throw new Error('record.harness.id and record.harness.name are required strings');
  }
  if (record.harness.version !== null && typeof record.harness.version !== 'string') {
    throw new Error('record.harness.version must be a string or null');
  }
  if (!record.workload || typeof record.workload !== 'object' || typeof record.workload.id !== 'string' && record.workload.id !== null) {
    throw new Error('record.workload.id is required as string or null');
  }
  if (!record.model || typeof record.model !== 'object' || (record.model.id !== null && typeof record.model.id !== 'string')) {
    throw new Error('record.model.id is required as string or null');
  }
  if (!record.metrics || typeof record.metrics !== 'object' || Array.isArray(record.metrics) || Object.keys(record.metrics).length === 0) {
    throw new Error('record.metrics is required and must contain at least one metric');
  }
  for (const [metricName, metricValue] of Object.entries(record.metrics)) {
    if (!(typeof metricValue === 'number' || metricValue === null)) {
      throw new Error(`record.metrics.${metricName} must be number or null`);
    }
  }
  if (!record.source || typeof record.source !== 'object' || typeof record.source.mode !== 'string') {
    throw new Error('record.source.mode is required');
  }
  if (record.source.commandTimeoutMs != null && (!Number.isInteger(record.source.commandTimeoutMs) || record.source.commandTimeoutMs <= 0)) {
    throw new Error('record.source.commandTimeoutMs must be a positive integer when present');
  }
  if (record.source.command !== null && !Array.isArray(record.source.command)) {
    throw new Error('record.source.command must be an array of strings or null');
  }
  if (record.source.command && !record.source.command.every((value) => typeof value === 'string')) {
    throw new Error('record.source.command must be an array of strings');
  }
  if (!record.environment || typeof record.environment !== 'object' || Array.isArray(record.environment)) {
    throw new Error('record.environment is required and must be an object');
  }
  for (const groupName of ['host', 'browser', 'gpu', 'runtime']) {
    const group = record.environment[groupName];
    if (!group || typeof group !== 'object' || Array.isArray(group)) {
      throw new Error(`record.environment.${groupName} must be an object`);
    }
  }
  if (record.notes !== null && typeof record.notes !== 'string') {
    throw new Error('record.notes must be a string or null');
  }
}

function toIsoTimestamp(timestamp = null) {
  return timestamp || new Date().toISOString();
}

function toFileTimestamp(timestamp = null) {
  return toIsoTimestamp(timestamp)
    .replace(/[:]/g, '-')
    .replace(/\.\d{3}Z$/, 'Z');
}

async function normalizeRecord(options) {
  const {
    product,
    harness,
    rawResult,
    workloadsById,
    workloadId,
    modelId,
    notes,
    source,
    resourceTelemetry,
    timestamp,
  } = options;

  const metricPaths = harness.normalization.metricPaths;
  const metrics = {};
  for (const [metricName, paths] of Object.entries(metricPaths)) {
    metrics[metricName] = firstFiniteNumber(rawResult, paths);
  }

  const missingRequired = [];
  for (const metricName of harness.normalization.requiredMetrics) {
    if (metrics[metricName] === null) {
      missingRequired.push(metricName);
    }
  }
  if (missingRequired.length > 0) {
    throw new Error(`Missing required metrics for ${product.id}: ${missingRequired.join(', ')}`);
  }

  const metadata = {};
  const metadataPaths = harness.normalization.metadataPaths || {};
  for (const [key, paths] of Object.entries(metadataPaths)) {
    metadata[key] = firstScalar(rawResult, paths);
  }
  const environment = buildNormalizedEnvironment(rawResult, harness, source);

  const selectedWorkload = workloadId ? workloadsById.get(workloadId) || null : null;
  if (workloadId && !selectedWorkload) {
    throw new Error(`Unknown workload id: ${workloadId}`);
  }

  const resolvedModelId = modelId || (metadata.model == null ? null : String(metadata.model));
  const harnessVersion = metadata.version == null ? null : String(metadata.version);

  const record = {
    schemaVersion: 1,
    timestamp: toIsoTimestamp(timestamp),
    target: {
      id: product.id,
      name: product.name,
      category: product.category,
      website: product.website,
    },
    harness: {
      id: harness.id,
      name: harness.name,
      version: harnessVersion,
    },
    workload: {
      id: selectedWorkload?.id ?? workloadId ?? null,
      name: selectedWorkload?.name ?? null,
    },
    model: {
      id: resolvedModelId,
      name: null,
    },
    metrics,
    source,
    environment,
    notes: notes ?? null,
    metadata,
  };
  if (resourceTelemetry != null) {
    record.resourceTelemetry = resourceTelemetry;
  }
  if (source?.mode === 'run') {
    assertRunEnvironmentCompleteness(environment, {
      requireGpu: String(product?.category || '').toLowerCase().includes('webgpu'),
    });
  }
  await assertMatchesSchema(record, RESULT_SCHEMA_PATH, 'result record');
  assertResultRecordShape(record);
  return record;
}

async function writeRecord(record, outputPath) {
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, `${JSON.stringify(record, null, 2)}\n`, 'utf8');
}

async function hashJsonBytes(value) {
  const bytes = Buffer.from(JSON.stringify(value), 'utf8');
  const digest = crypto.createHash('sha256').update(bytes).digest('hex');
  return { bytes: bytes.length, sha256: digest };
}

function hashTextBytes(text) {
  const bytes = Buffer.from(String(text), 'utf8');
  const digest = crypto.createHash('sha256').update(bytes).digest('hex');
  return { bytes: bytes.length, sha256: digest };
}

function resolveGitReleaseMetadata() {
  const commit = spawnSync('git', ['rev-parse', 'HEAD'], {
    cwd: ROOT_DIR,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'ignore'],
  });
  const commitSha = commit.status === 0
    ? asNonEmptyStringValue(commit.stdout)
    : null;

  const status = spawnSync('git', ['status', '--porcelain'], {
    cwd: ROOT_DIR,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'ignore'],
  });
  const dirty = status.status === 0
    ? status.stdout.trim().length > 0
    : null;

  return {
    commitSha,
    dirty,
  };
}

function defaultOutputPath(targetId, timestamp = null) {
  return path.join(RESULTS_DIR, `${targetId}-${toFileTimestamp(timestamp)}.json`);
}

async function loadTargetBundle(targetId, registry) {
  const product = registry.products.find((item) => item.id === targetId);
  if (!product) {
    throw new Error(`Unknown target "${targetId}".`);
  }
  const harnessPath = path.join(REGISTRY_DIR, product.harness);
  const harnessExists = await fileExists(harnessPath);
  if (!harnessExists) {
    throw new Error(`Harness file not found: ${product.harness}`);
  }
  const harness = await readJson(harnessPath);
  await assertMatchesSchema(harness, HARNESS_SCHEMA_PATH, `harness ${harnessPath}`);
  assertHarnessShape(harness, product.id);
  return { product, harness };
}

function printList(registry) {
  console.log('id\tstatus\tcategory\tname');
  for (const product of registry.products) {
    console.log(`${product.id}\t${product.status}\t${product.category}\t${product.name}`);
  }
}

function countEnabledFeatures(features) {
  let count = 0;
  for (const value of Object.values(features || {})) {
    if (isCapabilityFeatureSupported(value)) count += 1;
  }
  return count;
}

function resolveCapabilityTarget(capabilities, targetId) {
  const target = capabilities.targets.find((entry) => entry.id === targetId);
  if (!target) {
    throw new Error(`Unknown capability target "${targetId}"`);
  }
  return target;
}

function listMissingFeatures(baseFeatures, targetFeatures) {
  const missing = [];
  const keys = new Set([...Object.keys(baseFeatures || {}), ...Object.keys(targetFeatures || {})]);
  for (const key of keys) {
    if (isCapabilityFeatureSupported(baseFeatures?.[key]) && !isCapabilityFeatureSupported(targetFeatures?.[key])) {
      missing.push(key);
    }
  }
  return missing.sort();
}

function listExtraFeatures(baseFeatures, targetFeatures) {
  const extra = [];
  const keys = new Set([...Object.keys(baseFeatures || {}), ...Object.keys(targetFeatures || {})]);
  for (const key of keys) {
    if (!isCapabilityFeatureSupported(baseFeatures?.[key]) && isCapabilityFeatureSupported(targetFeatures?.[key])) {
      extra.push(key);
    }
  }
  return extra.sort();
}

function supportStatus(value) {
  return normalizeCapabilityFeatureStatus(value) || CAPABILITY_FEATURE_STATUS.unknown;
}

function toPosixRelative(filePath) {
  return path.relative(ROOT_DIR, filePath).split(path.sep).join('/');
}

function normalizeCatalogModeToken(value) {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'run' || normalized === 'text') return 'run';
  if (normalized === 'translate' || normalized === 'translation') return 'translate';
  if (normalized === 'embedding' || normalized === 'embed') return 'embedding';
  if (normalized === 'diffusion' || normalized === 'image') return 'diffusion';
  if (normalized === 'energy') return 'energy';
  return null;
}

function normalizeCatalogModes(rawMode, rawModes) {
  const tokens = new Set();
  const values = [];
  if (Array.isArray(rawModes)) values.push(...rawModes);
  if (rawMode !== undefined) values.push(rawMode);
  for (const value of values) {
    if (typeof value === 'string') {
      const lowered = value.trim().toLowerCase();
      if (lowered === 'both' || lowered === 'all' || lowered === 'text+embedding') {
        tokens.add('run');
        tokens.add('translate');
        tokens.add('embedding');
        continue;
      }
      const split = lowered.split(/[,\s+/]+/).filter(Boolean);
      for (const piece of split) {
        const normalized = normalizeCatalogModeToken(piece);
        if (normalized) tokens.add(normalized);
      }
      continue;
    }
    const normalized = normalizeCatalogModeToken(value);
    if (normalized) tokens.add(normalized);
  }
  if (tokens.has('run')) tokens.add('translate');
  if (tokens.has('translate')) tokens.add('run');
  if (tokens.size === 0) {
    tokens.add('run');
    tokens.add('translate');
  }
  return [...tokens].sort();
}

function normalizeCatalogTestedState(value) {
  const normalized = asNonEmptyString(value)?.toLowerCase() || null;
  if (normalized === 'verified' || normalized === 'pass' || normalized === 'passed') return 'verified';
  if (normalized === 'failed' || normalized === 'fail' || normalized === 'failing') return 'failed';
  return 'unknown';
}

function isCatalogEntryReleaseClaimable(entry) {
  if (!entry || typeof entry !== 'object' || Array.isArray(entry)) return false;
  const lifecycle = entry.lifecycle && typeof entry.lifecycle === 'object' ? entry.lifecycle : {};
  const status = lifecycle.status && typeof lifecycle.status === 'object' ? lifecycle.status : {};
  const tested = lifecycle.tested && typeof lifecycle.tested === 'object' ? lifecycle.tested : {};
  const runtimeStatus = asNonEmptyString(status.runtime)?.toLowerCase() || 'unknown';
  const testedStatus = normalizeCatalogTestedState(tested.result ?? status.tested);
  return runtimeStatus === 'active' && testedStatus === 'verified';
}

async function loadCompareConfigBundle() {
  const compareConfig = await readJson(COMPARE_CONFIG_PATH);
  await assertMatchesSchema(compareConfig, COMPARE_CONFIG_SCHEMA_PATH, 'compare-engines.config.json');
  if (!Array.isArray(compareConfig.modelProfiles)) {
    throw new Error('compare-engines.config.json modelProfiles must be an array');
  }
  return compareConfig;
}

async function loadCompareMetricBundle() {
  const raw = await fs.readFile(COMPARE_METRIC_CONTRACT_PATH, 'utf8');
  const compareMetricContract = JSON.parse(raw);
  await assertMatchesSchema(
    compareMetricContract,
    COMPARE_METRIC_CONTRACT_SCHEMA_PATH,
    'compare-metrics.json'
  );
  const metrics = Array.isArray(compareMetricContract?.metrics) ? compareMetricContract.metrics : [];
  return {
    path: toPosixRelative(COMPARE_METRIC_CONTRACT_PATH),
    sourceSha256: hashTextBytes(raw).sha256,
    metrics,
    metricIds: metrics
      .map((entry) => (typeof entry?.id === 'string' ? entry.id.trim() : ''))
      .filter(Boolean),
  };
}

function requireTrue(value, label) {
  if (value !== true) {
    throw new Error(`${label} must be true`);
  }
}

function assertKnownId(value, knownIds, label) {
  const normalized = asNonEmptyStringValue(value);
  if (!normalized || !knownIds.has(normalized)) {
    throw new Error(`${label} references unknown id: ${value}`);
  }
}

function repoPathToAbsolute(repoPath, label) {
  const normalized = asNonEmptyStringValue(repoPath);
  if (!normalized) {
    throw new Error(`${label} must be a non-empty repo-relative path`);
  }
  if (/^https?:\/\//i.test(normalized)) {
    throw new Error(`${label} must be repo-relative, got URL: ${normalized}`);
  }
  if (path.isAbsolute(normalized)) {
    throw new Error(`${label} must be repo-relative, got absolute path: ${normalized}`);
  }
  return path.resolve(ROOT_DIR, normalized);
}

async function assertRepoPathExists(repoPath, label) {
  const absolutePath = repoPathToAbsolute(repoPath, label);
  if (!(await fileExists(absolutePath))) {
    throw new Error(`${label} references missing file: ${repoPath}`);
  }
  return absolutePath;
}

function runtimeProfileIdToRepoPath(profileId) {
  const normalized = asNonEmptyStringValue(profileId)?.replace(/\.json$/u, '') || null;
  if (!normalized) return null;
  if (!normalized.startsWith('profiles/')) return null;
  return `src/config/runtime/profiles/${normalized.slice('profiles/'.length)}.json`;
}

async function hashRepoFileBytes(repoPath, label) {
  const absolutePath = await assertRepoPathExists(repoPath, label);
  const bytes = await fs.readFile(absolutePath);
  return crypto.createHash('sha256').update(bytes).digest('hex');
}

function computeShardSetHash(manifest) {
  const shards = Array.isArray(manifest?.shards) ? manifest.shards : [];
  if (shards.length === 0) return null;
  const shardHashInput = shards
    .map((shard) => {
      const filename = typeof shard?.filename === 'string' ? shard.filename : '';
      const size = Number.isFinite(Number(shard?.size)) ? Number(shard.size) : '';
      const hash = typeof shard?.hash === 'string'
        ? shard.hash
        : (typeof shard?.blake3 === 'string' ? shard.blake3 : '');
      return `${filename}:${size}:${hash}`;
    })
    .join('\n');
  return `sha256:${crypto.createHash('sha256').update(shardHashInput).digest('hex')}`;
}

function rawCatalogEntryByModelId(catalog, modelId) {
  const entries = Array.isArray(catalog?.entries) ? catalog.entries : [];
  return entries.find((entry) => entry?.modelId === modelId) || null;
}

function normalizeCatalogRuntimeStatus(entry) {
  return asNonEmptyStringValue(entry?.lifecycle?.status?.runtime)?.toLowerCase() || null;
}

function normalizeCatalogAvailabilityLocal(entry) {
  return entry?.lifecycle?.availability?.local === true;
}

function normalizeCatalogVerifiedExecution(entry) {
  const tested = entry?.lifecycle?.tested;
  const testedStatus = normalizeCatalogTestedState(tested?.result ?? entry?.lifecycle?.status?.tested);
  return testedStatus === 'verified';
}

function assertCompareEvidenceMetricCoverage(compareReport, metricIds, label) {
  if (!Array.isArray(metricIds) || metricIds.length === 0) return;
  const section = resolveCompareSection(compareReport);
  const dopplerPayload = resolveCompareEnginePayload(section?.payload, 'doppler');
  const tjsPayload = resolveCompareEnginePayload(section?.payload, 'transformersjs');
  if (!section?.payload || !dopplerPayload || !tjsPayload) {
    throw new Error(`${label} must contain a comparable Doppler and Transformers.js section`);
  }

  const missing = [];
  for (const metricId of metricIds) {
    if (readCompareMetric(dopplerPayload, metricId) == null) {
      missing.push(`doppler.${metricId}`);
    }
    if (readCompareMetric(tjsPayload, metricId) == null) {
      missing.push(`transformersjs.${metricId}`);
    }
  }
  if (missing.length > 0) {
    throw new Error(`${label} is missing required compare metrics: ${missing.join(', ')}`);
  }
}

function assertCompareEvidenceRequiredMeasurements(compareReport, requiredMeasurements, label) {
  const required = new Set(Array.isArray(requiredMeasurements) ? requiredMeasurements : []);
  const section = resolveCompareSection(compareReport);
  const dopplerPayload = resolveCompareEnginePayload(section?.payload, 'doppler');
  const tjsPayload = resolveCompareEnginePayload(section?.payload, 'transformersjs');
  if (!section?.payload || !dopplerPayload || !tjsPayload) {
    throw new Error(`${label} must contain a comparable Doppler and Transformers.js section`);
  }

  const missing = [];
  for (const metricId of ['peakMemoryBytes', 'residentMemoryBytes']) {
    if (!required.has(metricId)) continue;
    if (readCompareMetric(dopplerPayload, metricId) == null) {
      missing.push(`doppler.${metricId}`);
    }
    if (readCompareMetric(tjsPayload, metricId) == null) {
      missing.push(`transformersjs.${metricId}`);
    }
  }
  if (required.has('qualityParity')) {
    if (!compareReportSatisfiesOutputPolicy(compareReport)) {
      missing.push('qualityParity');
    }
  }
  if (required.has('failureCount')) {
    if (dopplerPayload.failed === true) missing.push('doppler.failureCount');
    if (tjsPayload.failed === true) missing.push('transformersjs.failureCount');
  }
  if (missing.length > 0) {
    throw new Error(`${label} is missing required claim measurements: ${missing.join(', ')}`);
  }
}

function resolveClaimCompareSection(compareReport, decodeProfileId) {
  if (!isJsonObject(compareReport?.sections)) return null;
  const computeSection = isJsonObject(compareReport.sections.compute)
    ? compareReport.sections.compute
    : null;
  if (computeSection && isJsonObject(computeSection[decodeProfileId])) {
    return computeSection[decodeProfileId];
  }
  if (
    asNonEmptyStringValue(compareReport.decodeProfile) === decodeProfileId
    && isJsonObject(compareReport.sections.warm)
  ) {
    return compareReport.sections.warm;
  }
  if (
    asNonEmptyStringValue(compareReport.decodeProfile) === decodeProfileId
    && isJsonObject(compareReport.sections.compute)
    && hasComparableSectionPayload(compareReport.sections.compute)
  ) {
    return compareReport.sections.compute;
  }
  return null;
}

function resolvePrimaryFairnessSection(compareReport) {
  const primarySection = asNonEmptyStringValue(compareReport?.fairness?.primarySection);
  if (primarySection) {
    const parts = primarySection.split('/');
    if (parts[0] === 'compute' && parts[1]) {
      const section = compareReport?.sections?.compute?.[parts[1]];
      if (isJsonObject(section)) return section;
    }
    const section = compareReport?.sections?.[primarySection];
    if (isJsonObject(section)) return section;
  }
  return resolveCompareSection(compareReport)?.payload || null;
}

function compareReportSatisfiesOutputPolicy(compareReport) {
  const exactMatch = compareReport?.correctness?.exactMatch === true;
  const normalizedMatch = compareReport?.correctness?.normalizedMatch === true;
  if (exactMatch || normalizedMatch) {
    return true;
  }
  const reportPolicy = compareReport?.methodology?.outputParity || {};
  const section = resolvePrimaryFairnessSection(compareReport);
  const sectionPolicy = section?.outputParityPolicy || {};
  return (
    compareReport?.fairness?.claimGrade === true
    && compareReport?.fairness?.correctnessOk === true
    && reportPolicy.requireMatch === false
    && reportPolicy.matchMode === 'decode-valid'
    && sectionPolicy.requireMatch === false
    && sectionPolicy.matchMode === 'decode-valid'
    && section?.pairedComparable === true
    && section?.invalidReason == null
    && section?.decodeValidity?.ok === true
  );
}

function assertClaimCompareScalar(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label} must be ${expected}, got ${actual}`);
  }
}

function assertClaimCompareNumber(actual, expected, label) {
  const parsedActual = asFiniteNumber(actual);
  const parsedExpected = asFiniteNumber(expected);
  if (parsedActual == null || parsedExpected == null || parsedActual !== parsedExpected) {
    throw new Error(`${label} must be ${expected}, got ${actual}`);
  }
}

function assertClaimCompareSection(
  section,
  decodeProfile,
  sharedRunContract,
  workload,
  label
) {
  if (!isJsonObject(section)) {
    throw new Error(`${label} must contain sections.compute.${decodeProfile.id}`);
  }
  if (section.pairedComparable !== true) {
    throw new Error(`${label} sections.compute.${decodeProfile.id}.pairedComparable must be true`);
  }
  if (section.invalidReason != null) {
    throw new Error(`${label} sections.compute.${decodeProfile.id}.invalidReason must be null`);
  }
  if (!resolveCompareEnginePayload(section, 'doppler')) {
    throw new Error(`${label} sections.compute.${decodeProfile.id} must contain Doppler payload`);
  }
  if (!resolveCompareEnginePayload(section, 'transformersjs')) {
    throw new Error(`${label} sections.compute.${decodeProfile.id} must contain Transformers.js payload`);
  }

  const promptTokens = section.promptTokens || {};
  assertClaimCompareScalar(
    promptTokens.ok,
    true,
    `${label} sections.compute.${decodeProfile.id}.promptTokens.ok`
  );
  assertClaimCompareScalar(
    promptTokens.pairedComparable,
    true,
    `${label} sections.compute.${decodeProfile.id}.promptTokens.pairedComparable`
  );
  if (workload) {
    assertClaimCompareNumber(
      promptTokens.target,
      workload.prefillTokens,
      `${label} sections.compute.${decodeProfile.id}.promptTokens.target`
    );
  }

  const decodeValidity = section.decodeValidity || {};
  assertClaimCompareScalar(
    decodeValidity.ok,
    true,
    `${label} sections.compute.${decodeProfile.id}.decodeValidity.ok`
  );
  for (const engineId of ['doppler', 'transformersjs']) {
    const decodeTokens = asFiniteNumber(decodeValidity?.[engineId]?.decodeTokens);
    if (decodeTokens == null || decodeTokens <= 0) {
      throw new Error(`${label} sections.compute.${decodeProfile.id}.decodeValidity.${engineId}.decodeTokens must be positive`);
    }
    if (decodeValidity?.[engineId]?.generatedTextEmpty === true) {
      throw new Error(`${label} sections.compute.${decodeProfile.id}.decodeValidity.${engineId}.generatedTextEmpty must not be true`);
    }
  }

  const cadence = section.dopplerDecodeCadence || {};
  assertClaimCompareNumber(
    cadence.batchSize,
    decodeProfile.batchSize,
    `${label} sections.compute.${decodeProfile.id}.dopplerDecodeCadence.batchSize`
  );
  assertClaimCompareNumber(
    cadence.readbackInterval,
    decodeProfile.readbackInterval,
    `${label} sections.compute.${decodeProfile.id}.dopplerDecodeCadence.readbackInterval`
  );
  assertClaimCompareScalar(
    cadence.disableMultiTokenDecode,
    decodeProfile.disableMultiTokenDecode,
    `${label} sections.compute.${decodeProfile.id}.dopplerDecodeCadence.disableMultiTokenDecode`
  );
  assertClaimCompareScalar(
    cadence.stopCheckMode,
    decodeProfile.stopCheckMode,
    `${label} sections.compute.${decodeProfile.id}.dopplerDecodeCadence.stopCheckMode`
  );

  const accounting = section.dopplerBatchAccounting || {};
  if (sharedRunContract?.promotionGates?.throughputCadence?.requireBatchAccounting === true) {
    assertClaimCompareScalar(
      accounting.schemaVersion,
      1,
      `${label} sections.compute.${decodeProfile.id}.dopplerBatchAccounting.schemaVersion`
    );
  }
  if (accounting.batchResolutionEfficiency != null) {
    const efficiency = asFiniteNumber(accounting.batchResolutionEfficiency);
    if (efficiency == null || efficiency < 0 || efficiency > 1) {
      throw new Error(`${label} sections.compute.${decodeProfile.id}.dopplerBatchAccounting.batchResolutionEfficiency must be between 0 and 1`);
    }
  }
  if (accounting.batchOverrunTokens != null) {
    const overrun = asFiniteNumber(accounting.batchOverrunTokens);
    if (overrun == null || overrun < 0) {
      throw new Error(`${label} sections.compute.${decodeProfile.id}.dopplerBatchAccounting.batchOverrunTokens must be non-negative`);
    }
  }

  const loadModes = new Set(sharedRunContract?.memoryResidency?.loadModes || []);
  if (loadModes.size > 0) {
    assertKnownId(section.loadMode, loadModes, `${label} sections.compute.${decodeProfile.id}.loadMode`);
  }
  const cacheModes = new Set(sharedRunContract?.memoryResidency?.cacheModes || []);
  if (cacheModes.size > 0) {
    assertKnownId(section.cacheMode, cacheModes, `${label} sections.compute.${decodeProfile.id}.cacheMode`);
  }
}

function assertClaimCompareThroughputGate(compareReport, matrix, label) {
  const gate = compareReport?.sections?.compute?.throughputCadenceGate;
  if (!isJsonObject(gate)) {
    throw new Error(`${label} sections.compute.throughputCadenceGate is required when throughput decode profile is cited`);
  }
  if (gate.ok !== true) {
    throw new Error(`${label} sections.compute.throughputCadenceGate.ok must be true`);
  }
  const thresholds = gate.thresholds || {};
  const expected = matrix.promotionGates?.throughputCadence || {};
  for (const field of [
    'requireBatchAccounting',
    'minDecodeTokensPerSecRatioVsParity',
    'minBatchResolutionEfficiency',
    'maxBatchOverrunTokens',
  ]) {
    if (thresholds[field] !== expected[field]) {
      throw new Error(`${label} sections.compute.throughputCadenceGate.thresholds.${field} must match local-inference-claim-matrix`);
    }
  }
}

function assertClaimMatrixCompareEvidence(compareReport, lane, matrix, workloads, label, evidenceLabel = null) {
  const compareEvidenceLabel = evidenceLabel || `${label}.evidence.compareResult`;
  const workloadId = asNonEmptyStringValue(compareReport?.workload?.id);
  const laneWorkloads = new Set(lane?.run?.workloads || []);
  if (!workloadId || !laneWorkloads.has(workloadId)) {
    throw new Error(`${compareEvidenceLabel} workload.id must be one of lane.run.workloads`);
  }
  const workload = (workloads?.workloads || []).find((entry) => entry?.id === workloadId) || null;
  if (!workload) {
    throw new Error(`${compareEvidenceLabel} workload.id references unknown workload: ${workloadId}`);
  }

  assertClaimCompareScalar(
    compareReport.dopplerModelId,
    lane.model.dopplerModelId,
    `${compareEvidenceLabel} dopplerModelId`
  );
  const tjsCompetitor = (lane.compare?.competitors || [])
    .find((competitor) => competitor?.target === 'transformersjs') || null;
  if (tjsCompetitor) {
    assertClaimCompareScalar(
      compareReport.tjsModelId,
      tjsCompetitor.modelId,
      `${compareEvidenceLabel} tjsModelId`
    );
  }

  const manifestSha256 = asNonEmptyStringValue(compareReport?.dopplerManifestPreflight?.manifestSha256)
    || asNonEmptyStringValue(compareReport?.dopplerModelSource?.manifestSha256);
  assertClaimCompareScalar(
    manifestSha256,
    lane.artifact.manifestSha256,
    `${compareEvidenceLabel} manifestSha256`
  );
  const manifestSource = asNonEmptyStringValue(compareReport?.dopplerModelSource?.manifestSource);
  if (manifestSource && manifestSource !== lane.artifact.manifestPath) {
    throw new Error(`${compareEvidenceLabel} manifestSource must match artifact.manifestPath`);
  }

  assertClaimCompareNumber(
    compareReport.maxTokens,
    workload.decodeTokens,
    `${compareEvidenceLabel} maxTokens`
  );
  assertClaimCompareNumber(
    compareReport.warmupRuns,
    matrix.sharedRunContract.warmupRuns,
    `${compareEvidenceLabel} warmupRuns`
  );
  const runs = asFiniteNumber(compareReport.runs);
  if (runs == null || runs < matrix.sharedRunContract.minTimedSamplesForPercentiles) {
    throw new Error(`${compareEvidenceLabel} runs must satisfy minTimedSamplesForPercentiles`);
  }
  assertClaimCompareNumber(
    compareReport.seed,
    matrix.sharedRunContract.samplingPolicy.seed,
    `${compareEvidenceLabel} seed`
  );
  const deterministic = compareReport?.methodology?.deterministicDecoding || {};
  assertClaimCompareNumber(
    deterministic.temperature,
    matrix.sharedRunContract.samplingPolicy.temperature,
    `${compareEvidenceLabel} methodology.deterministicDecoding.temperature`
  );
  assertClaimCompareNumber(
    deterministic.topK,
    matrix.sharedRunContract.samplingPolicy.topK,
    `${compareEvidenceLabel} methodology.deterministicDecoding.topK`
  );
  assertClaimCompareNumber(
    deterministic.topP,
    matrix.sharedRunContract.samplingPolicy.topP,
    `${compareEvidenceLabel} methodology.deterministicDecoding.topP`
  );

  if (compareReport?.correctness?.status === 'match') {
    assertClaimCompareScalar(
      compareReport?.correctness?.exactMatch,
      true,
      `${compareEvidenceLabel} correctness.exactMatch`
    );
    assertClaimCompareScalar(
      compareReport?.correctness?.normalizedMatch,
      true,
      `${compareEvidenceLabel} correctness.normalizedMatch`
    );
  } else if (!compareReportSatisfiesOutputPolicy(compareReport)) {
    throw new Error(
      `${compareEvidenceLabel} correctness.status must be match unless outputParityPolicy explicitly allows decode-valid product-format comparison`
    );
  }

  const decodeProfileById = new Map(
    (matrix.sharedRunContract.batchDecodeProfiles || [])
      .map((profile) => [profile.id, profile])
  );
  for (const decodeProfileId of lane.run?.decodeProfiles || []) {
    const decodeProfile = decodeProfileById.get(decodeProfileId);
    if (!decodeProfile) {
      throw new Error(`${label}.run.decodeProfiles references unknown decode profile: ${decodeProfileId}`);
    }
    const section = resolveClaimCompareSection(compareReport, decodeProfileId);
    assertClaimCompareSection(
      section,
      decodeProfile,
      {
        ...matrix.sharedRunContract,
        promotionGates: matrix.promotionGates,
      },
      workload,
      compareEvidenceLabel
    );
  }
  if ((lane.run?.decodeProfiles || []).includes('throughput')) {
    assertClaimCompareThroughputGate(compareReport, matrix, compareEvidenceLabel);
  }
}

function getLaneRequiredBackendIds(lane, matrix) {
  const laneBackends = lane?.run?.runtimeBackends || [];
  const requiredBackends = matrix?.sharedRunContract?.requiredRuntimeBackends || [];
  return requiredBackends
    .filter((requiredBackend) => laneBackends.some((runtimeBackend) => (
      runtimeBackendMatchesRequired(runtimeBackend, requiredBackend)
    )))
    .map((requiredBackend) => requiredBackend.id);
}

async function assertClaimSurfaceCompareResult(
  entry,
  lane,
  matrix,
  workloads,
  label,
  compareMetricIds,
  requiredMeasurements,
  requiredBackendById,
  laneRequiredBackendIds,
  options = {}
) {
  const backendId = asNonEmptyStringValue(entry?.backendId);
  const evidenceCollection = asNonEmptyStringValue(options.evidenceCollection) || 'surfaceCompareResults';
  const evidenceKey = asNonEmptyStringValue(options.evidenceKey) || backendId || 'unknown';
  const evidenceLabel = `${label}.evidence.${evidenceCollection}.${evidenceKey}`;
  const requiredBackend = requiredBackendById.get(backendId);
  if (!requiredBackend) {
    throw new Error(`${evidenceLabel}.backendId must reference sharedRunContract.requiredRuntimeBackends`);
  }
  if (!laneRequiredBackendIds.has(backendId)) {
    throw new Error(`${evidenceLabel}.backendId is not declared by lane.run.runtimeBackends`);
  }

  const compareResultPath = await assertRepoPathExists(entry.compareResult, `${evidenceLabel}.compareResult`);
  const compareReport = await readJson(compareResultPath);
  await assertCompareArtifactContracts(compareReport, entry.compareResult);
  assertCompareEvidenceMetricCoverage(compareReport, compareMetricIds, `${evidenceLabel}.compareResult`);
  assertCompareEvidenceRequiredMeasurements(compareReport, requiredMeasurements, `${evidenceLabel}.compareResult`);
  assertClaimMatrixCompareEvidence(compareReport, lane, matrix, workloads, label, `${evidenceLabel}.compareResult`);

  const actualSurface = asNonEmptyStringValue(compareReport?.dopplerSurface);
  if (actualSurface !== requiredBackend.surface) {
    throw new Error(
      `${evidenceLabel}.compareResult dopplerSurface must be ${requiredBackend.surface}, got ${actualSurface || 'missing'}`
    );
  }

  const expectedWorkloadId = asNonEmptyStringValue(options.expectedWorkloadId);
  if (expectedWorkloadId) {
    const actualWorkloadId = asNonEmptyStringValue(compareReport?.workload?.id);
    if (actualWorkloadId !== expectedWorkloadId) {
      throw new Error(
        `${evidenceLabel}.compareResult workload.id must be ${expectedWorkloadId}, got ${actualWorkloadId || 'missing'}`
      );
    }
  }

  if (entry.summarySvg != null) {
    await assertRepoPathExists(entry.summarySvg, `${evidenceLabel}.summarySvg`);
  } else if (options.requireSummarySvg === true) {
    throw new Error(`${evidenceLabel}.summarySvg is required`);
  }
}

async function assertClaimMatrixEvidence(lane, matrix, workloads, label, compareMetricIds, requiredMeasurements) {
  const evidence = lane?.evidence || {};
  if (evidence.localExecutionReport != null) {
    await assertRepoPathExists(evidence.localExecutionReport, `${label}.evidence.localExecutionReport`);
  }
  if (evidence.compareResult != null) {
    const compareResultPath = await assertRepoPathExists(evidence.compareResult, `${label}.evidence.compareResult`);
    const compareReport = await readJson(compareResultPath);
    await assertCompareArtifactContracts(compareReport, evidence.compareResult);
    assertCompareEvidenceMetricCoverage(
      compareReport,
      compareMetricIds,
      `${label}.evidence.compareResult`
    );
    assertCompareEvidenceRequiredMeasurements(
      compareReport,
      requiredMeasurements,
      `${label}.evidence.compareResult`
    );
    assertClaimMatrixCompareEvidence(compareReport, lane, matrix, workloads, label);
  }
  if (evidence.summarySvg != null) {
    await assertRepoPathExists(evidence.summarySvg, `${label}.evidence.summarySvg`);
  }

  const surfaceCompareResults = Array.isArray(evidence.surfaceCompareResults)
    ? evidence.surfaceCompareResults
    : [];
  const workloadCompareResults = Array.isArray(evidence.workloadCompareResults)
    ? evidence.workloadCompareResults
    : [];
  const requiredBackendById = new Map(
    (matrix.sharedRunContract?.requiredRuntimeBackends || []).map((backend) => [backend.id, backend])
  );
  const laneRequiredBackendIds = new Set(getLaneRequiredBackendIds(lane, matrix));
  const laneWorkloadIds = new Set(Array.isArray(lane?.run?.workloads) ? lane.run.workloads : []);
  const seenBackendIds = new Set();
  for (const entry of surfaceCompareResults) {
    const backendId = asNonEmptyStringValue(entry?.backendId);
    if (seenBackendIds.has(backendId)) {
      throw new Error(`${label}.evidence.surfaceCompareResults contains duplicate backendId ${backendId}`);
    }
    seenBackendIds.add(backendId);
    await assertClaimSurfaceCompareResult(
      entry,
      lane,
      matrix,
      workloads,
      label,
      compareMetricIds,
      requiredMeasurements,
      requiredBackendById,
      laneRequiredBackendIds,
      {
        evidenceCollection: 'surfaceCompareResults',
        evidenceKey: backendId,
        requireSummarySvg: true,
      }
    );
  }
  const seenBackendWorkloadIds = new Set();
  for (const entry of workloadCompareResults) {
    const backendId = asNonEmptyStringValue(entry?.backendId);
    const workloadId = asNonEmptyStringValue(entry?.workloadId);
    const evidenceKey = `${backendId || 'unknown'}:${workloadId || 'unknown'}`;
    if (!laneWorkloadIds.has(workloadId)) {
      throw new Error(`${label}.evidence.workloadCompareResults.${evidenceKey}.workloadId is not declared by lane.run.workloads`);
    }
    if (seenBackendWorkloadIds.has(evidenceKey)) {
      throw new Error(`${label}.evidence.workloadCompareResults contains duplicate backend/workload ${evidenceKey}`);
    }
    seenBackendWorkloadIds.add(evidenceKey);
    await assertClaimSurfaceCompareResult(
      entry,
      lane,
      matrix,
      workloads,
      label,
      compareMetricIds,
      requiredMeasurements,
      requiredBackendById,
      laneRequiredBackendIds,
      {
        evidenceCollection: 'workloadCompareResults',
        evidenceKey,
        expectedWorkloadId: workloadId,
        requireSummarySvg: false,
      }
    );
  }
  if (evidence.compareResult != null && laneRequiredBackendIds.size > 1) {
    for (const backendId of laneRequiredBackendIds) {
      if (!seenBackendIds.has(backendId)) {
        throw new Error(`${label}.evidence.surfaceCompareResults is missing backendId ${backendId}`);
      }
    }
  }
}

function assertRequiredRuntimeBackends(matrix, context, label) {
  const requiredBackends = matrix.sharedRunContract?.requiredRuntimeBackends || [];
  const productIds = context.productIds;
  const capabilities = context.capabilities;
  const benchFeatureIds = new Set(
    (capabilities?.featureCatalog?.bench || [])
      .map((entry) => asNonEmptyStringValue(entry?.id))
      .filter(Boolean)
  );
  const targetById = new Map(
    (capabilities?.targets || [])
      .map((target) => [asNonEmptyStringValue(target?.id), target])
      .filter(([targetId]) => Boolean(targetId))
  );
  collectUniqueIds(
    requiredBackends.map((backend) => backend.id),
    `${label} sharedRunContract.requiredRuntimeBackends`
  );
  for (const backend of requiredBackends) {
    const backendLabel = `${label} sharedRunContract.requiredRuntimeBackends.${backend.id}`;
    assertKnownId(backend.target, productIds, `${backendLabel}.target`);
    if (!benchFeatureIds.has(backend.feature)) {
      throw new Error(`${backendLabel}.feature references unknown bench capability: ${backend.feature}`);
    }
    const target = targetById.get(backend.target);
    if (!target) {
      throw new Error(`${backendLabel}.target has no capabilities entry: ${backend.target}`);
    }
    if (!isCapabilityFeatureSupported(target.bench?.features?.[backend.feature])) {
      throw new Error(`${backendLabel}.feature is not supported by capabilities target ${backend.target}`);
    }
  }
}

function runtimeBackendMatchesRequired(candidate, required) {
  return candidate?.target === required.target
    && candidate?.surface === required.surface
    && candidate?.backend === required.backend
    && candidate?.format === required.format;
}

function assertPromotedRuntimeBackendCoverage(lane, requiredBackends, label) {
  if (lane.status !== 'promoted') return;
  const laneBackends = Array.isArray(lane.run?.runtimeBackends) ? lane.run.runtimeBackends : [];
  for (const required of requiredBackends) {
    if (required.requiredForPromotion !== true) continue;
    const matches = laneBackends.some((backend) => runtimeBackendMatchesRequired(backend, required));
    if (!matches) {
      throw new Error(
        `${label} is promoted but run.runtimeBackends is missing required coverage `
        + `${required.id} (${required.target}/${required.surface}/${required.backend}/${required.format})`
      );
    }
  }
}

async function assertClaimMatrixArtifact(lane, catalogEntry, label) {
  const manifestPath = lane?.artifact?.manifestPath;
  const manifestHash = await hashRepoFileBytes(manifestPath, `${label}.artifact.manifestPath`);
  if (manifestHash !== lane?.artifact?.manifestSha256) {
    throw new Error(
      `${label}.artifact.manifestSha256 mismatch for ${manifestPath} `
      + `(expected ${lane.artifact.manifestSha256}, current ${manifestHash})`
    );
  }

  const manifest = await readJson(repoPathToAbsolute(manifestPath, `${label}.artifact.manifestPath`));
  const manifestModelId = asNonEmptyStringValue(manifest?.modelId);
  if (manifestModelId !== lane?.model?.dopplerModelId) {
    throw new Error(`${label}.artifact.manifestPath modelId mismatch: expected ${lane.model.dopplerModelId}, got ${manifestModelId}`);
  }

  const tokenizerFile = asNonEmptyStringValue(lane?.model?.tokenizer?.file);
  if (tokenizerFile) {
    const tokenizerPath = path.posix.join(path.posix.dirname(manifestPath), tokenizerFile);
    const tokenizerHash = await hashRepoFileBytes(tokenizerPath, `${label}.model.tokenizer.file`);
    if (tokenizerHash !== lane?.model?.tokenizer?.sha256) {
      throw new Error(
        `${label}.model.tokenizer.sha256 mismatch for ${tokenizerPath} `
        + `(expected ${lane.model.tokenizer.sha256}, current ${tokenizerHash})`
      );
    }
  }

  const identity = manifest?.artifactIdentity || {};
  const expectedSourceCheckpoint = asNonEmptyStringValue(identity.sourceCheckpointId);
  if (expectedSourceCheckpoint && lane.model.sourceCheckpointId !== expectedSourceCheckpoint) {
    throw new Error(`${label}.model.sourceCheckpointId must match manifest artifactIdentity.sourceCheckpointId`);
  }
  const expectedSourceRevision = asNonEmptyStringValue(identity.sourceRevision);
  if (expectedSourceRevision && lane.model.sourceRevision !== expectedSourceRevision) {
    throw new Error(`${label}.model.sourceRevision must match manifest artifactIdentity.sourceRevision`);
  }
  const expectedWeightPackId = asNonEmptyStringValue(identity.weightPackId);
  if (expectedWeightPackId && lane.artifact.weightPackId !== expectedWeightPackId) {
    throw new Error(`${label}.artifact.weightPackId must match manifest artifactIdentity.weightPackId`);
  }
  const expectedWeightPackHash = asNonEmptyStringValue(identity.weightPackHash);
  if (expectedWeightPackHash && lane.artifact.weightPackHash !== expectedWeightPackHash) {
    throw new Error(`${label}.artifact.weightPackHash must match manifest artifactIdentity.weightPackHash`);
  }
  const expectedManifestVariantId = asNonEmptyStringValue(identity.manifestVariantId);
  if (expectedManifestVariantId && lane.artifact.manifestVariantId !== expectedManifestVariantId) {
    throw new Error(`${label}.artifact.manifestVariantId must match manifest artifactIdentity.manifestVariantId`);
  }
  const expectedShardSetHash = asNonEmptyStringValue(identity.shardSetHash) || computeShardSetHash(manifest);
  if (expectedShardSetHash && lane.artifact.shardSetHash !== expectedShardSetHash) {
    throw new Error(`${label}.artifact.shardSetHash must match manifest shard identity`);
  }

  const catalogSizeBytes = Number(catalogEntry?.sizeBytes);
  if (Number.isFinite(catalogSizeBytes) && lane.artifact.sizeBytes !== catalogSizeBytes) {
    throw new Error(`${label}.artifact.sizeBytes must match models/catalog.json sizeBytes`);
  }
}

async function assertLocalInferenceClaimMatrixShape(matrix, context) {
  const {
    registry,
    workloads,
    compareConfig,
    compareMetricBundle,
    benchmarkPolicy,
    catalog,
    capabilities,
  } = context;
  if (matrix.schemaVersion !== 1) {
    throw new Error(`local-inference-claim-matrix schemaVersion must be 1, got ${matrix.schemaVersion}`);
  }

  const productIds = new Set((registry.products || []).map((entry) => entry.id));
  const workloadIds = new Set((workloads.workloads || []).map((entry) => entry.id));
  const compareProfileByModelId = new Map(
    (compareConfig.modelProfiles || []).map((profile) => [profile.dopplerModelId, profile])
  );
  const decodeProfileIds = new Set((matrix.sharedRunContract.batchDecodeProfiles || []).map((profile) => profile.id));
  const benchmarkDecodeProfiles = benchmarkPolicy?.decodeProfiles?.profiles || {};
  const allowedRuntimeStatuses = new Set(matrix.selectionPolicy.allowedRuntimeStatuses || []);
  const allowedFormats = new Set(matrix.selectionPolicy.allowedCompressionFormats || []);
  const requiredRuntimeBackends = matrix.sharedRunContract?.requiredRuntimeBackends || [];

  for (const requiredGate of [
    'nonZeroExecution',
    'providerImportsOk',
    'noCpuFallback',
    'promptSemanticsMatch',
    'outputSemanticsMatch',
    'stableReadbackAccounting',
    'percentilesPresent',
    'artifactHashesMatch',
    'memoryBudgetRespected',
    'compareArtifactRequired',
  ]) {
    requireTrue(matrix.promotionGates?.[requiredGate], `local-inference-claim-matrix promotionGates.${requiredGate}`);
  }
  const matrixThroughputGate = matrix.promotionGates?.throughputCadence;
  if (!isJsonObject(matrixThroughputGate)) {
    throw new Error('local-inference-claim-matrix promotionGates.throughputCadence must be an object');
  }
  const benchmarkThroughputGate = benchmarkPolicy?.promotionGates?.throughputCadence;
  if (!isJsonObject(benchmarkThroughputGate)) {
    throw new Error('benchmark-policy promotionGates.throughputCadence must be an object');
  }
  for (const field of [
    'requireBatchAccounting',
    'minDecodeTokensPerSecRatioVsParity',
    'minBatchResolutionEfficiency',
    'maxBatchOverrunTokens',
  ]) {
    if (matrixThroughputGate[field] !== benchmarkThroughputGate[field]) {
      throw new Error(`local-inference-claim-matrix promotionGates.throughputCadence.${field} must match benchmark-policy promotionGates.throughputCadence.${field}`);
    }
  }
  requireTrue(matrix.sharedRunContract.memoryResidency?.forbidCpuFallback, 'local-inference-claim-matrix sharedRunContract.memoryResidency.forbidCpuFallback');
  requireTrue(matrix.sharedRunContract.memoryResidency?.recordPeakMemory, 'local-inference-claim-matrix sharedRunContract.memoryResidency.recordPeakMemory');
  requireTrue(matrix.sharedRunContract.memoryResidency?.recordResidentMemory, 'local-inference-claim-matrix sharedRunContract.memoryResidency.recordResidentMemory');
  requireTrue(matrix.sharedRunContract.competitorPolicy?.samePrompts, 'local-inference-claim-matrix sharedRunContract.competitorPolicy.samePrompts');
  requireTrue(matrix.sharedRunContract.competitorPolicy?.sameSampling, 'local-inference-claim-matrix sharedRunContract.competitorPolicy.sameSampling');
  requireTrue(matrix.sharedRunContract.competitorPolicy?.sameHardware, 'local-inference-claim-matrix sharedRunContract.competitorPolicy.sameHardware');
  requireTrue(matrix.sharedRunContract.competitorPolicy?.sameReportShape, 'local-inference-claim-matrix sharedRunContract.competitorPolicy.sameReportShape');

  for (const workloadId of matrix.sharedRunContract.workloads || []) {
    assertKnownId(workloadId, workloadIds, `local-inference-claim-matrix sharedRunContract.workloads`);
  }
  for (const targetId of matrix.sharedRunContract.competitorPolicy.targets || []) {
    assertKnownId(targetId, productIds, `local-inference-claim-matrix sharedRunContract.competitorPolicy.targets`);
  }
  for (const profile of matrix.sharedRunContract.batchDecodeProfiles || []) {
    const label = `local-inference-claim-matrix sharedRunContract.batchDecodeProfiles.${profile.id}`;
    const benchmarkProfile = benchmarkDecodeProfiles[profile.id];
    if (!benchmarkProfile || typeof benchmarkProfile !== 'object' || Array.isArray(benchmarkProfile)) {
      throw new Error(`${label} must exist in benchmark-policy decodeProfiles.profiles`);
    }
    for (const field of ['batchSize', 'readbackInterval', 'disableMultiTokenDecode', 'stopCheckMode']) {
      if (profile[field] !== benchmarkProfile[field]) {
        throw new Error(`${label}.${field} must match benchmark-policy decodeProfiles.profiles.${profile.id}.${field}`);
      }
    }
  }
  assertRequiredRuntimeBackends(matrix, { productIds, capabilities }, 'local-inference-claim-matrix');
  for (const metricId of compareMetricBundle.metricIds || []) {
    if (!matrix.requiredMeasurements.includes(metricId)) {
      throw new Error(`local-inference-claim-matrix requiredMeasurements must include compare metric "${metricId}"`);
    }
  }

  const laneIds = matrix.lanes.map((lane) => lane.id);
  collectUniqueIds(laneIds, 'local inference claim lane');

  for (const [index, lane] of matrix.lanes.entries()) {
    const label = `local-inference-claim-matrix lanes[${index}] (${lane.id})`;
    const dopplerModelId = lane.model.dopplerModelId;
    const catalogEntry = rawCatalogEntryByModelId(catalog, dopplerModelId);
    if (!catalogEntry) {
      throw new Error(`${label} references model absent from models/catalog.json: ${dopplerModelId}`);
    }
    if (matrix.selectionPolicy.requiresLocalManifest && !normalizeCatalogAvailabilityLocal(catalogEntry)) {
      throw new Error(`${label} requires local catalog availability`);
    }
    const runtimeStatus = normalizeCatalogRuntimeStatus(catalogEntry);
    if (!allowedRuntimeStatuses.has(runtimeStatus)) {
      throw new Error(`${label} runtime status "${runtimeStatus}" is not allowed by selectionPolicy`);
    }
    if (matrix.selectionPolicy.requiresVerifiedLocalExecution && !normalizeCatalogVerifiedExecution(catalogEntry)) {
      throw new Error(`${label} requires verified local execution in models/catalog.json`);
    }
    if (lane.artifact.sizeBytes > matrix.selectionPolicy.maxArtifactBytes) {
      throw new Error(`${label} artifact size exceeds selectionPolicy.maxArtifactBytes`);
    }
    if (lane.memoryBudget.artifactBytes !== lane.artifact.sizeBytes) {
      throw new Error(`${label}.memoryBudget.artifactBytes must match artifact.sizeBytes`);
    }
    if (lane.memoryBudget.maxResidentBytes < lane.artifact.sizeBytes) {
      throw new Error(`${label}.memoryBudget.maxResidentBytes must be at least artifact.sizeBytes`);
    }
    if (!allowedFormats.has(lane.artifact.format)) {
      throw new Error(`${label}.artifact.format is not allowed by selectionPolicy`);
    }

    const compareProfile = compareProfileByModelId.get(dopplerModelId);
    if (matrix.selectionPolicy.requiresCompareProfile && !compareProfile) {
      throw new Error(`${label} requires compare-engines.config.json profile`);
    }
    await assertClaimMatrixArtifact(lane, catalogEntry, label);
    await assertClaimMatrixEvidence(
      lane,
      matrix,
      workloads,
      label,
      compareMetricBundle.metricIds || [],
      matrix.requiredMeasurements || []
    );

    for (const workloadId of lane.run.workloads || []) {
      assertKnownId(workloadId, workloadIds, `${label}.run.workloads`);
    }
    for (const decodeProfileId of lane.run.decodeProfiles || []) {
      assertKnownId(decodeProfileId, decodeProfileIds, `${label}.run.decodeProfiles`);
    }
    const laneRuntimeProfiles = lane.run.runtimeProfileByDecodeProfile || {};
    for (const [decodeProfileId, runtimeProfileId] of Object.entries(laneRuntimeProfiles)) {
      assertKnownId(decodeProfileId, decodeProfileIds, `${label}.run.runtimeProfileByDecodeProfile`);
      if (!(lane.run.decodeProfiles || []).includes(decodeProfileId)) {
        throw new Error(`${label}.run.runtimeProfileByDecodeProfile.${decodeProfileId} must also appear in run.decodeProfiles`);
      }
      if (runtimeProfileId == null) continue;
      const runtimeProfilePath = runtimeProfileIdToRepoPath(runtimeProfileId);
      if (!runtimeProfilePath) {
        throw new Error(`${label}.run.runtimeProfileByDecodeProfile.${decodeProfileId} must be a profiles/* runtime profile id`);
      }
      await assertRepoPathExists(
        runtimeProfilePath,
        `${label}.run.runtimeProfileByDecodeProfile.${decodeProfileId}`
      );
      const compareRuntimeProfileId = compareProfile?.dopplerRuntimeProfileByDecodeProfile?.[decodeProfileId] ?? null;
      if (compareRuntimeProfileId !== runtimeProfileId) {
        throw new Error(
          `${label}.run.runtimeProfileByDecodeProfile.${decodeProfileId} must match compare-engines.config.json`
        );
      }
    }
    for (const backend of lane.run.runtimeBackends || []) {
      assertKnownId(backend.target, productIds, `${label}.run.runtimeBackends.target`);
    }
    assertPromotedRuntimeBackendCoverage(lane, requiredRuntimeBackends, label);
    for (const competitor of lane.compare.competitors || []) {
      assertKnownId(competitor.target, productIds, `${label}.compare.competitors.target`);
      if (competitor.target === 'transformersjs' && compareProfile?.defaultTjsModelId) {
        if (competitor.modelId !== compareProfile.defaultTjsModelId) {
          throw new Error(`${label}.compare.competitors modelId must match compare-engines.config.json defaultTjsModelId`);
        }
      }
    }

    if (lane.status === 'promoted') {
      if (!asNonEmptyStringValue(lane.evidence.compareResult)) {
        throw new Error(`${label} is promoted but evidence.compareResult is missing`);
      }
      if (!asNonEmptyStringValue(lane.evidence.summarySvg)) {
        throw new Error(`${label} is promoted but evidence.summarySvg is missing`);
      }
      for (const competitor of lane.compare.competitors || []) {
        if (!asNonEmptyStringValue(competitor.expectedArtifactHash)) {
          throw new Error(`${label} is promoted but competitor ${competitor.target} expectedArtifactHash is missing`);
        }
      }
    }
  }
}

async function loadLocalInferenceClaimMatrixBundle(context) {
  const matrix = await readJson(LOCAL_INFERENCE_CLAIM_MATRIX_PATH);
  await assertMatchesSchema(
    matrix,
    LOCAL_INFERENCE_CLAIM_MATRIX_SCHEMA_PATH,
    'local-inference-claim-matrix.json'
  );
  await assertLocalInferenceClaimMatrixShape(matrix, context);
  return matrix;
}

async function loadModelCatalogBundle() {
  const catalog = await readJson(MODEL_CATALOG_PATH);
  const entries = Array.isArray(catalog?.models) ? catalog.models : [];
  const byModelId = new Map();
  for (const entry of entries) {
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) continue;
    const modelId = typeof entry.modelId === 'string' ? entry.modelId.trim() : '';
    if (!modelId) continue;
    byModelId.set(modelId, {
      modelId,
      label: typeof entry.label === 'string' ? entry.label : modelId,
      description: typeof entry.description === 'string' ? entry.description : '',
      baseUrl: typeof entry.baseUrl === 'string' ? entry.baseUrl : null,
      sizeBytes: Number.isFinite(Number(entry.sizeBytes)) ? Number(entry.sizeBytes) : null,
      recommended: entry.recommended === true,
      sortOrder: Number.isFinite(Number(entry.sortOrder)) ? Number(entry.sortOrder) : null,
      modes: normalizeCatalogModes(entry.mode, entry.modes),
      releaseClaimable: isCatalogEntryReleaseClaimable(entry),
    });
  }
  return {
    updatedAt: typeof catalog?.updatedAt === 'string' ? catalog.updatedAt : null,
    entries,
    byModelId,
  };
}

function hasComparableSectionPayload(section) {
  if (!section || typeof section !== 'object' || Array.isArray(section)) return false;
  return Boolean(section?.doppler || section?.transformersjs || section?.result || section?.tjs);
}

function resolveCompareSection(report) {
  if (!report || typeof report !== 'object' || Array.isArray(report)) return null;
  const sections = report.sections;
  if (!sections || typeof sections !== 'object') return null;
  const computeSections = isJsonObject(sections.compute) ? sections.compute : null;
  const throughputSection = computeSections?.throughput;
  if (
    computeSections?.throughputCadenceGate?.ok === true
    && hasComparableSectionPayload(throughputSection)
    && throughputSection?.pairedComparable !== false
  ) {
    return {
      id: 'compute/throughput',
      payload: throughputSection,
    };
  }
  const primarySection = asNonEmptyStringValue(report?.fairness?.primarySection);
  if (primarySection) {
    const primaryChain = primarySection.split('/').filter(Boolean);
    let cursor = sections;
    for (const segment of primaryChain) {
      if (!cursor || typeof cursor !== 'object') {
        cursor = null;
        break;
      }
      cursor = cursor[segment];
    }
    if (hasComparableSectionPayload(cursor) && cursor?.pairedComparable !== false) {
      return {
        id: primarySection,
        payload: cursor,
      };
    }
  }
  const candidates = [
    ['compute', 'throughput'],
    ['compute', 'parity'],
    ['warm'],
    ['cold'],
  ];
  let fallback = null;
  for (const chain of candidates) {
    let cursor = sections;
    for (const segment of chain) {
      if (!cursor || typeof cursor !== 'object') {
        cursor = null;
        break;
      }
      cursor = cursor[segment];
    }
    if (hasComparableSectionPayload(cursor)) {
      const section = {
        id: chain.join('/'),
        payload: cursor,
      };
      if (cursor?.pairedComparable !== false) {
        return section;
      }
      if (fallback == null) {
        fallback = section;
      }
    }
  }
  return fallback;
}

function resolveCompareEnginePayload(section, engineId) {
  if (!section || typeof section !== 'object' || Array.isArray(section)) return null;
  if (engineId === 'transformersjs') {
    return section.transformersjs ?? section.tjs ?? null;
  }
  return section.doppler ?? section.result ?? null;
}

function asFiniteNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function asOptionalFiniteNumber(value) {
  if (value === null || value === undefined || value === '') return null;
  return asFiniteNumber(value);
}

function readCompareMetric(payload, metricId) {
  if (!payload || typeof payload !== 'object') return null;
  const metricPaths = [
    ['result', 'timing', metricId],
    ['result', 'metrics', metricId],
    ['timing', metricId],
    ['metrics', metricId],
    ['result', metricId],
    [metricId],
  ];
  if (metricId === 'promptTokensPerSecToFirstToken') {
    metricPaths.push(
      ['result', 'metrics', 'medianPrefillTokensPerSecTtft'],
      ['result', 'metrics', 'avgPrefillTokensPerSecTtft'],
      ['result', 'timing', 'prefillTokensPerSecTtft'],
      ['result', 'timing', 'prefillTokensPerSec'],
      ['metrics', 'medianPrefillTokensPerSecTtft'],
      ['metrics', 'avgPrefillTokensPerSecTtft'],
      ['timing', 'prefillTokensPerSecTtft'],
      ['timing', 'prefillTokensPerSec']
    );
  }
  if (metricId === 'peakMemoryBytes') {
    metricPaths.push(
      ['result', 'memoryStats', 'pool', 'peakBytesAllocated'],
      ['result', 'memoryStats', 'pool', 'peakBytesRequested'],
      ['result', 'memoryStats', 'used'],
      ['memoryStats', 'pool', 'peakBytesAllocated'],
      ['memoryStats', 'pool', 'peakBytesRequested'],
      ['memoryStats', 'used'],
      ['memoryInfo', 'after', 'usedJSHeapSize'],
      ['memoryInfo', 'before', 'usedJSHeapSize'],
      ['result', 'memoryInfo', 'after', 'usedJSHeapSize'],
      ['result', 'memoryInfo', 'before', 'usedJSHeapSize']
    );
  }
  if (metricId === 'residentMemoryBytes') {
    metricPaths.push(
      ['result', 'memoryStats', 'pool', 'currentBytesAllocated'],
      ['result', 'memoryStats', 'pool', 'currentBytesRequested'],
      ['result', 'memoryStats', 'used'],
      ['memoryStats', 'pool', 'currentBytesAllocated'],
      ['memoryStats', 'pool', 'currentBytesRequested'],
      ['memoryStats', 'used'],
      ['memoryInfo', 'after', 'usedJSHeapSize'],
      ['memoryInfo', 'before', 'usedJSHeapSize'],
      ['result', 'memoryInfo', 'after', 'usedJSHeapSize'],
      ['result', 'memoryInfo', 'before', 'usedJSHeapSize']
    );
  }
  for (const pathParts of metricPaths) {
    let cursor = payload;
    for (const key of pathParts) {
      if (!cursor || typeof cursor !== 'object') {
        cursor = undefined;
        break;
      }
      cursor = cursor[key];
    }
    const value = asFiniteNumber(cursor);
    if (value != null) return value;
  }
  return null;
}

function summarizeDopplerBottleneck(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
  const dominant = raw.dominant && typeof raw.dominant === 'object' && !Array.isArray(raw.dominant)
    ? raw.dominant
    : {};
  const dominantId = asNonEmptyString(dominant.id);
  const dominantMs = asOptionalFiniteNumber(dominant.ms);
  if (!dominantId && dominantMs == null) return null;

  const components = raw.componentsMs && typeof raw.componentsMs === 'object' && !Array.isArray(raw.componentsMs)
    ? raw.componentsMs
    : {};
  const recording = raw.recording && typeof raw.recording === 'object' && !Array.isArray(raw.recording)
    ? raw.recording
    : {};
  const topOps = Array.isArray(recording.topOps)
    ? recording.topOps.slice(0, 8).map((entry) => ({
        label: asNonEmptyString(entry?.label) || null,
        count: asOptionalFiniteNumber(entry?.count),
        shareOfOps: asOptionalFiniteNumber(entry?.shareOfOps),
      })).filter((entry) => entry.label || entry.count != null)
    : [];

  return {
    schemaVersion: Number.isInteger(raw.schemaVersion) ? raw.schemaVersion : 1,
    bottleneckClass: asNonEmptyString(raw.bottleneckClass) || null,
    dominant: {
      id: dominantId || null,
      label: asNonEmptyString(dominant.label) || null,
      ms: dominantMs,
      shareOfDecode: asOptionalFiniteNumber(dominant.shareOfDecode),
    },
    decodeWallMs: asOptionalFiniteNumber(raw.decodeWallMs),
    componentsMs: {
      commandRecordMs: asOptionalFiniteNumber(components.commandRecordMs),
      submitWaitMs: asOptionalFiniteNumber(components.submitWaitMs),
      readbackWaitMs: asOptionalFiniteNumber(components.readbackWaitMs),
      effectiveSubmitReadbackWaitMs: asOptionalFiniteNumber(components.effectiveSubmitReadbackWaitMs),
      readbackMapWaitMs: asOptionalFiniteNumber(components.readbackMapWaitMs),
      readbackCleanupMs: asOptionalFiniteNumber(components.readbackCleanupMs),
      readbackCopyMs: asOptionalFiniteNumber(components.readbackCopyMs),
      gpuTimestampMs: asOptionalFiniteNumber(components.gpuTimestampMs),
      orchestrationMs: asOptionalFiniteNumber(components.orchestrationMs),
      residualMs: asOptionalFiniteNumber(components.residualMs),
    },
    recording: {
      opCount: asOptionalFiniteNumber(recording.opCount),
      passCount: asOptionalFiniteNumber(recording.passCount),
      uniqueOpLabels: asOptionalFiniteNumber(recording.uniqueOpLabels),
      msPerOp: asOptionalFiniteNumber(recording.msPerOp),
      msPerPass: asOptionalFiniteNumber(recording.msPerPass),
      opsPerExecutedBatchToken: asOptionalFiniteNumber(recording.opsPerExecutedBatchToken),
      passesPerExecutedBatchToken: asOptionalFiniteNumber(recording.passesPerExecutedBatchToken),
      topOps,
    },
  };
}

function firstStringByPaths(payload, dottedPaths) {
  if (!payload || typeof payload !== 'object') return null;
  for (const dottedPath of dottedPaths || []) {
    const candidate = getByPath(payload, dottedPath);
    if (typeof candidate === 'string' && candidate.trim() !== '') {
      return candidate.trim();
    }
  }
  return null;
}

function firstBooleanByPaths(payload, dottedPaths) {
  if (!payload || typeof payload !== 'object') return null;
  for (const dottedPath of dottedPaths || []) {
    const candidate = getByPath(payload, dottedPath);
    if (typeof candidate === 'boolean') {
      return candidate;
    }
    if (typeof candidate === 'string') {
      const normalized = candidate.trim().toLowerCase();
      if (normalized === 'true') return true;
      if (normalized === 'false') return false;
    }
  }
  return null;
}

function firstNonEmptyString(...values) {
  for (const value of values) {
    if (typeof value === 'string' && value.trim() !== '') {
      return value.trim();
    }
  }
  return null;
}

function firstDefinedBoolean(...values) {
  for (const value of values) {
    if (typeof value === 'boolean') return value;
  }
  return null;
}

function firstKnownString(...values) {
  for (const value of values) {
    if (typeof value !== 'string') continue;
    const trimmed = value.trim();
    if (trimmed === '') continue;
    const normalized = trimmed.toLowerCase();
    if (normalized === 'unknown' || normalized === 'n/a' || normalized === 'na') continue;
    if (trimmed === '0x0000') continue;
    return trimmed;
  }
  return null;
}

function inferRuntimeFallbackFromArtifactPath(repoPath) {
  const fileName = path.basename(String(repoPath || '')).toLowerCase();
  if (fileName.includes('.apple-m3pro.')) {
    return {
      host: {
        platform: 'darwin',
        cpuModel: 'Apple M3',
      },
      browser: {
        executable: 'chromium',
      },
      gpu: {
        backend: 'metal',
        vendor: 'apple',
        description: 'Apple M3',
      },
    };
  }
  return null;
}

function summarizeCompareEngineEnvironment(payload, engineId) {
  const library = firstNonEmptyString(
    firstStringByPaths(payload, ['env.library', 'result.env.library', 'result.result.env.library']),
    engineId === 'doppler' ? 'doppler' : null,
    engineId === 'transformersjs' ? 'transformers.js' : null,
  );
  const version = firstStringByPaths(payload, ['env.version', 'result.env.version', 'result.result.env.version']);
  const surface = firstStringByPaths(payload, ['surface', 'result.surface', 'result.result.surface']);
  const browserUserAgent = firstStringByPaths(payload, [
    'env.browserUserAgent',
    'env.browser.userAgent',
    'result.env.browserUserAgent',
    'result.env.browser.userAgent',
    'result.result.env.browserUserAgent',
    'result.result.env.browser.userAgent',
  ]);
  const browserPlatform = firstStringByPaths(payload, [
    'env.browserPlatform',
    'env.browser.platform',
    'result.env.browserPlatform',
    'result.env.browser.platform',
    'result.result.env.browserPlatform',
    'result.result.env.browser.platform',
  ]);
  const browserLanguage = firstStringByPaths(payload, [
    'env.browserLanguage',
    'env.browser.language',
    'result.env.browserLanguage',
    'result.env.browser.language',
    'result.result.env.browserLanguage',
    'result.result.env.browser.language',
  ]);
  const browserVendor = firstStringByPaths(payload, [
    'env.browserVendor',
    'env.browser.vendor',
    'result.env.browserVendor',
    'result.env.browser.vendor',
    'result.result.env.browserVendor',
    'result.result.env.browser.vendor',
  ]);
  const browserExecutable = firstStringByPaths(payload, [
    'environment.browser.executable',
    'result.environment.browser.executable',
    'result.result.environment.browser.executable',
  ]);

  const gpuVendor = firstStringByPaths(payload, [
    'deviceInfo.adapterInfo.vendor',
    'result.deviceInfo.adapterInfo.vendor',
    'result.result.deviceInfo.adapterInfo.vendor',
    'deviceInfo.vendor',
    'result.deviceInfo.vendor',
    'result.result.deviceInfo.vendor',
  ]);
  const gpuArchitecture = firstStringByPaths(payload, [
    'deviceInfo.adapterInfo.architecture',
    'result.deviceInfo.adapterInfo.architecture',
    'result.result.deviceInfo.adapterInfo.architecture',
    'deviceInfo.architecture',
    'result.deviceInfo.architecture',
    'result.result.deviceInfo.architecture',
  ]);
  const gpuDevice = firstStringByPaths(payload, [
    'deviceInfo.adapterInfo.device',
    'result.deviceInfo.adapterInfo.device',
    'result.result.deviceInfo.adapterInfo.device',
    'deviceInfo.device',
    'result.deviceInfo.device',
    'result.result.deviceInfo.device',
  ]);
  const gpuDescription = firstStringByPaths(payload, [
    'deviceInfo.adapterInfo.description',
    'result.deviceInfo.adapterInfo.description',
    'result.result.deviceInfo.adapterInfo.description',
    'deviceInfo.description',
    'result.deviceInfo.description',
    'result.result.deviceInfo.description',
  ]);
  const gpuBackend = firstStringByPaths(payload, [
    'env.webgpuBackend',
    'env.gpuBackend',
    'env.graphicsBackend',
    'result.env.webgpuBackend',
    'result.env.gpuBackend',
    'result.env.graphicsBackend',
    'result.result.env.webgpuBackend',
    'result.result.env.gpuBackend',
    'result.result.env.graphicsBackend',
    'environment.gpu.backend',
    'result.environment.gpu.backend',
    'result.result.environment.gpu.backend',
  ]);
  const hasF16 = firstBooleanByPaths(payload, [
    'deviceInfo.hasF16',
    'result.deviceInfo.hasF16',
    'result.result.deviceInfo.hasF16',
  ]);
  const hasSubgroups = firstBooleanByPaths(payload, [
    'deviceInfo.hasSubgroups',
    'result.deviceInfo.hasSubgroups',
    'result.result.deviceInfo.hasSubgroups',
  ]);
  const hasTimestampQuery = firstBooleanByPaths(payload, [
    'deviceInfo.hasTimestampQuery',
    'result.deviceInfo.hasTimestampQuery',
    'result.result.deviceInfo.hasTimestampQuery',
  ]);

  return {
    library,
    version,
    surface,
    runtime: {
      device: firstStringByPaths(payload, [
        'env.device',
        'result.env.device',
        'result.result.env.device',
      ]),
      dtype: firstStringByPaths(payload, [
        'env.dtype',
        'result.env.dtype',
        'result.result.env.dtype',
      ]),
      requestedDtype: firstStringByPaths(payload, [
        'env.requestedDtype',
        'result.env.requestedDtype',
        'result.result.env.requestedDtype',
      ]),
      executionProviderMode: firstStringByPaths(payload, [
        'env.executionProviderMode',
        'result.env.executionProviderMode',
        'result.result.env.executionProviderMode',
      ]),
    },
    browser: {
      userAgent: browserUserAgent,
      platform: browserPlatform,
      language: browserLanguage,
      vendor: browserVendor,
      executable: browserExecutable,
    },
    gpu: {
      backend: gpuBackend,
      vendor: gpuVendor,
      architecture: gpuArchitecture,
      device: gpuDevice,
      description: gpuDescription,
      hasF16,
      hasSubgroups,
      hasTimestampQuery,
    },
  };
}

async function maybeLoadCompareResultSummary(
  compareResultPath,
  compareMetricIds = null,
  compareMetricDefinitions = null
) {
  if (!compareResultPath) return null;
  const resolved = path.resolve(compareResultPath);
  const repoPath = toPosixRelative(resolved);
  const exists = await fileExists(resolved);
  if (!exists) {
    throw new Error(`compare result not found: ${compareResultPath}`);
  }
  const report = await readJson(resolved);
  await assertCompareArtifactContracts(report, resolved);
  const runtimeFallback = inferRuntimeFallbackFromArtifactPath(repoPath);
  const section = resolveCompareSection(report);
  const dopplerPayload = resolveCompareEnginePayload(section?.payload, 'doppler');
  const tjsPayload = resolveCompareEnginePayload(section?.payload, 'transformersjs');
  const dopplerEnvironment = summarizeCompareEngineEnvironment(dopplerPayload, 'doppler');
  const tjsEnvironment = summarizeCompareEngineEnvironment(tjsPayload, 'transformersjs');
  const hostCpuModel = firstKnownString(
    firstStringByPaths(report, ['environment.host.cpuModel']),
    runtimeFallback?.host?.cpuModel
  );
  const hostEnvironment = {
    platform: firstKnownString(
      firstStringByPaths(report, ['environment.host.platform']),
      runtimeFallback?.host?.platform
    ),
    arch: firstKnownString(firstStringByPaths(report, ['environment.host.arch'])),
    nodeVersion: firstStringByPaths(report, ['environment.host.nodeVersion']),
  };
  const browserEnvironment = {
    userAgent: firstKnownString(
      tjsEnvironment.browser.userAgent,
      dopplerEnvironment.browser.userAgent,
    ),
    platform: firstKnownString(
      tjsEnvironment.browser.platform,
      dopplerEnvironment.browser.platform,
      firstStringByPaths(report, ['environment.browser.platform'])
    ),
    language: firstKnownString(
      tjsEnvironment.browser.language,
      dopplerEnvironment.browser.language,
    ),
    vendor: firstKnownString(
      tjsEnvironment.browser.vendor,
      dopplerEnvironment.browser.vendor,
    ),
    executable: firstKnownString(
      firstStringByPaths(report, ['environment.browser.executable']),
      tjsEnvironment.browser.executable,
      dopplerEnvironment.browser.executable,
      runtimeFallback?.browser?.executable,
    ),
  };
  const gpuEnvironment = {
    backend: firstKnownString(
      tjsEnvironment.gpu.backend,
      dopplerEnvironment.gpu.backend,
      firstStringByPaths(report, ['environment.gpu.backend']),
      runtimeFallback?.gpu?.backend
    ),
    vendor: firstKnownString(
      tjsEnvironment.gpu.vendor,
      dopplerEnvironment.gpu.vendor,
      firstStringByPaths(report, ['environment.gpu.vendor']),
      runtimeFallback?.gpu?.vendor
    ),
    architecture: firstKnownString(
      tjsEnvironment.gpu.architecture,
      dopplerEnvironment.gpu.architecture,
      firstStringByPaths(report, ['environment.gpu.architecture'])
    ),
    device: firstKnownString(
      tjsEnvironment.gpu.device,
      dopplerEnvironment.gpu.device,
      firstStringByPaths(report, ['environment.gpu.device'])
    ),
    description: firstKnownString(
      tjsEnvironment.gpu.description,
      dopplerEnvironment.gpu.description,
      firstStringByPaths(report, ['environment.gpu.description']),
      runtimeFallback?.gpu?.description,
      hostCpuModel
    ),
    hasF16: firstDefinedBoolean(tjsEnvironment.gpu.hasF16, dopplerEnvironment.gpu.hasF16),
    hasSubgroups: firstDefinedBoolean(tjsEnvironment.gpu.hasSubgroups, dopplerEnvironment.gpu.hasSubgroups),
    hasTimestampQuery: firstDefinedBoolean(tjsEnvironment.gpu.hasTimestampQuery, dopplerEnvironment.gpu.hasTimestampQuery),
  };
  if (
    runtimeFallback?.gpu?.description
    && typeof gpuEnvironment.description === 'string'
    && /^apple m3$/i.test(gpuEnvironment.description.trim())
  ) {
    gpuEnvironment.description = runtimeFallback.gpu.description;
  }
  const metricIds = Array.isArray(compareMetricIds) && compareMetricIds.length > 0
    ? compareMetricIds
    : ['decodeTokensPerSec', 'firstTokenMs', 'firstResponseMs', 'modelLoadMs'];
  const metrics = {};
  for (const metricId of metricIds) {
    metrics[metricId] = {
      doppler: readCompareMetric(dopplerPayload, metricId),
      transformersjs: readCompareMetric(tjsPayload, metricId),
    };
  }
  const selectedDecodeProfile = typeof section?.id === 'string' && section.id.startsWith('compute/')
    ? section.id.slice('compute/'.length)
    : (typeof report.decodeProfile === 'string' ? report.decodeProfile : null);
  return {
    path: repoPath,
    timestamp: typeof report.timestamp === 'string' ? report.timestamp : null,
    mode: typeof report.mode === 'string' ? report.mode : null,
    section: section?.id ?? null,
    cacheMode: asNonEmptyString(section?.payload?.cacheMode),
    loadMode: asNonEmptyString(section?.payload?.loadMode),
    pairedComparable: section?.payload?.pairedComparable !== false,
    invalidReason: typeof section?.payload?.invalidReason === 'string'
      ? section.payload.invalidReason
      : null,
    decodeProfile: selectedDecodeProfile,
    computeSectionIds: Object.keys(isJsonObject(report?.sections?.compute) ? report.sections.compute : {})
      .filter((sectionId) => sectionId !== 'throughputCadenceGate')
      .filter((sectionId) => isJsonObject(report.sections.compute[sectionId])),
    dopplerSurface: asNonEmptyString(report.dopplerSurface),
    dopplerExecution: {
      requestedSurface: asNonEmptyString(report?.dopplerExecution?.requestedSurface),
      commandSurface: asNonEmptyString(report?.dopplerExecution?.commandSurface),
      cliExecutor: asNonEmptyString(report?.dopplerExecution?.cliExecutor),
      commandSurfaceReason: asNonEmptyString(report?.dopplerExecution?.commandSurfaceReason),
    },
    dopplerModelId: typeof report.dopplerModelId === 'string' ? report.dopplerModelId : null,
    dopplerModelSource: typeof report?.dopplerModelSource?.source === 'string'
      ? report.dopplerModelSource.source
      : null,
    tjsModelId: typeof report.tjsModelId === 'string' ? report.tjsModelId : null,
    compareLane: typeof report?.compareLane?.declared === 'string'
      ? report.compareLane.declared
      : null,
    compareLaneReason: typeof report?.compareLane?.reason === 'string'
      ? report.compareLane.reason
      : null,
    fairness: isJsonObject(report?.fairness)
      ? {
          claimGrade: report.fairness.claimGrade === true,
          releaseClaimable: report.fairness.releaseClaimable === true,
          localComparable: report.fairness.localComparable === true,
          correctnessOk: report.fairness.correctnessOk === true,
          primarySection: asNonEmptyString(report.fairness.primarySection),
          invalidReason: asNonEmptyString(report.fairness.invalidReason),
          invalidReasons: Array.isArray(report.fairness.invalidReasons)
            ? report.fairness.invalidReasons.map((reason) => String(reason)).filter(Boolean)
            : [],
        }
      : null,
    dopplerKernelPath: typeof report.dopplerKernelPath === 'string' ? report.dopplerKernelPath : null,
    correctness: {
      status: asNonEmptyString(report?.correctness?.status) || null,
      exactMatch: report?.correctness?.exactMatch === true,
      normalizedMatch: report?.correctness?.normalizedMatch === true,
      matchingPrefixTokens: asFiniteNumber(report?.correctness?.tokenMatch?.matchingPrefixTokens),
      firstMismatchTokenIndex: asFiniteNumber(report?.correctness?.tokenMatch?.firstMismatchTokenIndex),
    },
    workloadId: typeof report?.workload?.id === 'string' ? report.workload.id : null,
    workload: report?.workload && typeof report.workload === 'object'
      ? {
          prefillTokenTarget: asFiniteNumber(report.workload.prefillTokenTarget),
          decodeTokenTarget: asFiniteNumber(report.workload.decodeTokenTarget),
        }
      : null,
    environment: {
      host: hostEnvironment,
      browser: browserEnvironment,
      gpu: gpuEnvironment,
      engines: {
        doppler: dopplerEnvironment,
        transformersjs: tjsEnvironment,
      },
    },
    metrics,
    dopplerBottleneck: summarizeDopplerBottleneck(section?.payload?.dopplerBottleneck),
    bottlenecks: buildCompareMetricBottlenecks(metrics, compareMetricDefinitions),
  };
}

function roundMetricRatio(value) {
  if (!Number.isFinite(value)) return null;
  return Math.round(value * 1_000_000) / 1_000_000;
}

function buildCompareMetricBottlenecks(metrics, compareMetricDefinitions) {
  if (!metrics || typeof metrics !== 'object' || Array.isArray(metrics)) return [];
  const definitions = Array.isArray(compareMetricDefinitions) ? compareMetricDefinitions : [];
  const rows = [];
  for (const definition of definitions) {
    const metricId = typeof definition?.id === 'string' ? definition.id.trim() : '';
    if (!metricId) continue;
    const metric = metrics[metricId];
    const doppler = asFiniteNumber(metric?.doppler);
    const transformersjs = asFiniteNumber(metric?.transformersjs);
    if (doppler == null || transformersjs == null) continue;
    const higherBetter = definition.higherBetter !== false;
    const tjsLeads = higherBetter
      ? transformersjs > doppler
      : transformersjs < doppler;
    if (!tjsLeads) continue;
    const denominator = Math.abs(higherBetter ? doppler : transformersjs);
    if (denominator === 0) continue;
    const gapRatio = higherBetter
      ? (transformersjs - doppler) / denominator
      : (doppler - transformersjs) / denominator;
    rows.push({
      metricId,
      label: typeof definition.label === 'string' ? definition.label : metricId,
      unit: typeof definition.unit === 'string' ? definition.unit : null,
      higherBetter,
      leader: 'transformersjs',
      doppler,
      transformersjs,
      gapRatio: roundMetricRatio(gapRatio),
      gapPercent: roundMetricRatio(gapRatio * 100),
    });
  }
  return rows
    .filter((row) => row.gapRatio != null && row.gapRatio > 0)
    .sort((left, right) => {
      const byGap = right.gapRatio - left.gapRatio;
      if (byGap !== 0) return byGap;
      return left.metricId.localeCompare(right.metricId);
    });
}

async function listCommittedCharts() {
  const exists = await fileExists(RESULTS_DIR);
  if (!exists) return [];
  const entries = await fs.readdir(RESULTS_DIR, { withFileTypes: true });
  return entries
    .filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith('.svg'))
    .map((entry) => toPosixRelative(path.join(RESULTS_DIR, entry.name)))
    .sort();
}

function isCommittedCompareFixtureFileName(fileName) {
  const lower = String(fileName || '').trim().toLowerCase();
  if (!lower.endsWith('.json')) return false;
  return lower.endsWith('.compare.json');
}

function isLocalCompareResultArtifactFileName(fileName) {
  const lower = String(fileName || '').trim().toLowerCase();
  if (!lower.endsWith('.json')) return false;
  if (lower === 'compare_latest.json') return false;
  return lower.startsWith('compare_') || lower.endsWith('.compare.json');
}

function isPathWithin(baseDir, candidatePath) {
  const relativePath = path.relative(baseDir, candidatePath);
  return relativePath !== ''
    && !relativePath.startsWith('..')
    && !path.isAbsolute(relativePath);
}

function resolveCompareArtifactOrigin(absolutePath) {
  if (isPathWithin(FIXTURES_DIR, absolutePath)) return 'fixture';
  if (isPathWithin(RESULTS_DIR, absolutePath)) return 'local';
  return 'explicit';
}

async function listCompareResultCandidateEntries(options = {}) {
  const includeLocalResults = options.includeLocalResults === true;
  const scanDirs = [{ dirPath: FIXTURES_DIR, origin: 'fixture' }];
  if (includeLocalResults) {
    scanDirs.push({ dirPath: RESULTS_DIR, origin: 'local' });
  }

  const out = [];
  for (const { dirPath, origin } of scanDirs) {
    const exists = await fileExists(dirPath);
    if (!exists) continue;
    const entries = await fs.readdir(dirPath, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isFile()) continue;
      if (origin === 'fixture' && !isCommittedCompareFixtureFileName(entry.name)) continue;
      if (origin === 'local' && !isLocalCompareResultArtifactFileName(entry.name)) continue;
      out.push({
        absolutePath: path.join(dirPath, entry.name),
        origin,
      });
    }
  }

  return out.sort((left, right) => left.absolutePath.localeCompare(right.absolutePath));
}

function compareResultTimestampMs(summary) {
  if (!summary || typeof summary !== 'object') return Number.NEGATIVE_INFINITY;
  const timestamp = typeof summary.timestamp === 'string' ? summary.timestamp : '';
  const ms = Date.parse(timestamp);
  if (!Number.isFinite(ms)) return Number.NEGATIVE_INFINITY;
  return ms;
}

function compareResultSortDescending(left, right) {
  const delta = compareResultTimestampMs(right) - compareResultTimestampMs(left);
  if (delta !== 0) return delta;
  const leftOrigin = String(left?.origin || '');
  const rightOrigin = String(right?.origin || '');
  if (leftOrigin !== rightOrigin) {
    if (leftOrigin === 'local') return -1;
    if (rightOrigin === 'local') return 1;
  }
  return String(left?.path || '').localeCompare(String(right?.path || ''));
}

function compareResultModelTypeKey(summary) {
  const dopplerModelId = asNonEmptyString(summary?.dopplerModelId);
  if (dopplerModelId) return `doppler:${dopplerModelId}`;
  const tjsModelId = asNonEmptyString(summary?.tjsModelId);
  if (tjsModelId) return `transformersjs:${tjsModelId}`;
  return 'unknown';
}

function compareResultRuntimeTypeKey(summary) {
  const host = summary?.environment?.host || {};
  const browser = summary?.environment?.browser || {};
  const gpu = summary?.environment?.gpu || {};
  const browserLabel = normalizeBrowserLabel(browser);
  const dopplerExecution = summary?.dopplerExecution || {};
  const values = [
    asNonEmptyString(host.platform) || 'unknown',
    asNonEmptyString(host.arch) || 'unknown',
    asNonEmptyString(summary?.dopplerSurface) || 'unknown',
    asNonEmptyString(dopplerExecution.commandSurface) || 'unknown',
    asNonEmptyString(dopplerExecution.cliExecutor) || 'unknown',
    browserLabel,
    asNonEmptyString(gpu.backend) || 'unknown',
    asNonEmptyString(gpu.vendor) || 'unknown',
    asNonEmptyString(gpu.architecture) || 'unknown',
    asNonEmptyString(gpu.device) || 'unknown',
    asNonEmptyString(gpu.description) || 'unknown',
  ];
  return values.join('|');
}

function compareResultUniqueTypeKey(summary) {
  const workloadId = asNonEmptyString(summary?.workloadId);
  if (workloadId) {
    return `${workloadId}::${compareResultModelTypeKey(summary)}::${compareResultRuntimeTypeKey(summary)}`;
  }
  return asNonEmptyString(summary?.path) || '';
}

function selectLatestCompareResultsByType(compareResults) {
  const sorted = Array.isArray(compareResults)
    ? [...compareResults].sort(compareResultSortDescending)
    : [];
  const selected = [];
  const seenKeys = new Set();
  for (const summary of sorted) {
    const key = compareResultUniqueTypeKey(summary);
    if (!key || seenKeys.has(key)) continue;
    seenKeys.add(key);
    selected.push(summary);
  }
  return selected;
}

function selectLatestCompareResult(compareResults) {
  const normalized = Array.isArray(compareResults) ? compareResults : [];
  if (normalized.length === 0) return null;
  const nonFixture = normalized.filter((entry) => entry?.origin !== 'fixture');
  const pool = nonFixture.length > 0 ? nonFixture : normalized;
  return [...pool].sort(compareResultSortDescending)[0] || null;
}

async function loadCompareResultSummaries(options = {}) {
  const compareResultFlag = options.compareResultFlag ?? null;
  const compareMetricIds = Array.isArray(options.compareMetricIds) ? options.compareMetricIds : null;
  const compareMetricDefinitions = Array.isArray(options.compareMetricDefinitions)
    ? options.compareMetricDefinitions
    : null;
  const includeLocalResults = options.includeLocalResults === true;
  const strictCompareArtifacts = options.strictCompareArtifacts === true;
  const candidateEntries = await listCompareResultCandidateEntries({ includeLocalResults });
  const candidateMap = new Map();
  for (const entry of candidateEntries) {
    candidateMap.set(path.resolve(entry.absolutePath), entry);
  }
  let preferredResolved = null;
  if (compareResultFlag) {
    preferredResolved = path.resolve(compareResultFlag);
    candidateMap.set(preferredResolved, {
      absolutePath: preferredResolved,
      origin: resolveCompareArtifactOrigin(preferredResolved),
      explicit: true,
    });
  }

  const summaries = [];
  const droppedCompareArtifacts = [];
  for (const candidateEntry of [...candidateMap.values()].sort((left, right) => left.absolutePath.localeCompare(right.absolutePath))) {
    const candidatePath = candidateEntry.absolutePath;
    try {
      const summary = await maybeLoadCompareResultSummary(
        candidatePath,
        compareMetricIds,
        compareMetricDefinitions
      );
      if (!summary) continue;
      summary.origin = candidateEntry.origin || resolveCompareArtifactOrigin(candidatePath);
      summaries.push(summary);
    } catch (error) {
      const isPreferred = preferredResolved != null && path.resolve(candidatePath) === preferredResolved;
      const droppedRow = {
        path: toPosixRelative(candidatePath),
        origin: candidateEntry.origin || resolveCompareArtifactOrigin(candidatePath),
        error: error?.message || String(error),
      };
      droppedCompareArtifacts.push(droppedRow);
      if (isPreferred) {
        throw error;
      }
    }
  }

  if (strictCompareArtifacts && droppedCompareArtifacts.length > 0) {
    const details = droppedCompareArtifacts
      .map((entry) => `- ${entry.path} (${entry.origin}): ${entry.error}`)
      .join('\n');
    throw new Error(`invalid compare artifacts detected:\n${details}`);
  }
  summaries.sort(compareResultSortDescending);

  let preferredSummary = null;
  if (preferredResolved != null) {
    const preferredRelativePath = toPosixRelative(preferredResolved);
    preferredSummary = summaries.find((entry) => entry.path === preferredRelativePath) || null;
  }

  return {
    compareResults: summaries,
    latestCompareResult: preferredSummary || selectLatestCompareResult(summaries),
    droppedCompareArtifacts,
  };
}

async function hashCompareArtifactSource(sourcePath) {
  if (typeof sourcePath !== 'string' || sourcePath.trim() === '') {
    throw new Error('compare artifact source path is required');
  }
  let absolutePath = sourcePath;
  const dopplerMarker = 'doppler/';
  const idx = sourcePath.indexOf(dopplerMarker);
  if (idx !== -1) {
    absolutePath = path.resolve(ROOT_DIR, sourcePath.slice(idx + dopplerMarker.length));
  } else if (!path.isAbsolute(sourcePath)) {
    absolutePath = path.resolve(ROOT_DIR, sourcePath);
  }
  if (!(await fileExists(absolutePath))) {
    throw new Error(`compare artifact source is missing: ${sourcePath} (resolved: ${absolutePath})`);
  }
  const raw = await fs.readFile(absolutePath, 'utf8');
  const hashInfo = hashTextBytes(raw);
  return {
    absolutePath,
    sha256: hashInfo.sha256,
  };
}

function normalizeCompareLaneForArtifact(value) {
  if (isJsonObject(value)) {
    return asNonEmptyStringValue(value.declared)
      || asNonEmptyStringValue(value.lane)
      || null;
  }
  return asNonEmptyStringValue(value);
}

async function assertCompatibleCompareConfigDrift(report, compareResultPath, currentSha256, expectedSha256) {
  const config = await readJson(COMPARE_CONFIG_PATH);
  const modelId = asNonEmptyStringValue(report?.dopplerModelId);
  const profiles = Array.isArray(config?.modelProfiles) ? config.modelProfiles : [];
  const profile = profiles.find((entry) => entry?.dopplerModelId === modelId) || null;
  if (!profile) {
    throw new Error(
      `compare artifact ${compareResultPath} has stale compareConfig hash `
      + `(expected ${expectedSha256}, current ${currentSha256}) and no current profile for ${modelId || 'unknown model'}`
    );
  }

  const mismatches = [];
  const reportLane = normalizeCompareLaneForArtifact(report?.compareLane);
  const profileLane = asNonEmptyStringValue(profile.compareLane);
  if (profileLane && reportLane && profileLane !== reportLane) {
    mismatches.push(`compareLane ${reportLane} -> ${profileLane}`);
  }

  const reportTjsModelId = asNonEmptyStringValue(report?.tjsModelId);
  const profileTjsModelId = asNonEmptyStringValue(profile.defaultTjsModelId);
  if (profileTjsModelId && reportTjsModelId && profileTjsModelId !== reportTjsModelId) {
    mismatches.push(`defaultTjsModelId ${reportTjsModelId} -> ${profileTjsModelId}`);
  }

  const reportFormats = isJsonObject(report?.methodology?.formats) ? report.methodology.formats : {};
  const reportDopplerFormat = asNonEmptyStringValue(reportFormats.doppler);
  const profileDopplerFormat = asNonEmptyStringValue(profile.defaultDopplerFormat);
  if (profileDopplerFormat && reportDopplerFormat && profileDopplerFormat !== reportDopplerFormat) {
    mismatches.push(`defaultDopplerFormat ${reportDopplerFormat} -> ${profileDopplerFormat}`);
  }

  const reportTjsFormat = asNonEmptyStringValue(reportFormats.transformersjs);
  const profileTjsFormat = asNonEmptyStringValue(profile.defaultTjsFormat);
  if (profileTjsFormat && reportTjsFormat && profileTjsFormat !== reportTjsFormat) {
    mismatches.push(`defaultTjsFormat ${reportTjsFormat} -> ${profileTjsFormat}`);
  }

  if (mismatches.length > 0) {
    throw new Error(
      `compare artifact ${compareResultPath} has stale compareConfig hash `
      + `(expected ${expectedSha256}, current ${currentSha256}) with profile drift: ${mismatches.join('; ')}`
    );
  }
}

async function assertCompareArtifactContracts(report, compareResultPath) {
  const checks = [
    {
      label: 'benchmarkPolicy',
      source: report?.benchmarkPolicy?.source,
      sourceSha256: report?.benchmarkPolicy?.sourceSha256,
    },
    {
      label: 'compareConfig',
      source: report?.compareConfig?.source,
      sourceSha256: report?.compareConfig?.sourceSha256,
    },
    {
      label: 'metricContract',
      source: report?.metricContract?.source,
      sourceSha256: report?.metricContract?.sourceSha256,
    },
    {
      label: 'dopplerHarness',
      source: report?.harnesses?.doppler?.source,
      sourceSha256: report?.harnesses?.doppler?.sourceSha256,
    },
    {
      label: 'transformersjsHarness',
      source: report?.harnesses?.transformersjs?.source,
      sourceSha256: report?.harnesses?.transformersjs?.sourceSha256,
    },
  ];
  for (const check of checks) {
    if (typeof check.source !== 'string' || check.source.trim() === '') {
      throw new Error(`compare artifact ${compareResultPath} is missing ${check.label}.source`);
    }
    if (typeof check.sourceSha256 !== 'string' || check.sourceSha256.trim() === '') {
      throw new Error(`compare artifact ${compareResultPath} is missing ${check.label}.sourceSha256`);
    }
    const current = await hashCompareArtifactSource(check.source);
    if (current.sha256 !== check.sourceSha256) {
      if (check.label === 'compareConfig') {
        await assertCompatibleCompareConfigDrift(
          report,
          compareResultPath,
          current.sha256,
          check.sourceSha256
        );
        continue;
      }
      throw new Error(
        `compare artifact ${compareResultPath} has stale ${check.label} hash `
        + `(expected ${check.sourceSha256}, current ${current.sha256})`
      );
    }
  }
}

function toMarkdownLinkHref(repoPath, markdownPath) {
  if (typeof repoPath !== 'string' || repoPath.trim() === '') return null;
  if (!markdownPath) return null;
  const absoluteTarget = path.resolve(ROOT_DIR, repoPath);
  const relativeHref = path.relative(path.dirname(markdownPath), absoluteTarget)
    .split(path.sep)
    .join('/');
  if (relativeHref === '') return './';
  if (relativeHref.startsWith('.')) return relativeHref;
  return `./${relativeHref}`;
}

function formatRepoPathLink(repoPath, markdownPath, label = null) {
  if (typeof repoPath !== 'string' || repoPath.trim() === '') return '';
  const href = toMarkdownLinkHref(repoPath, markdownPath);
  if (!href) return `\`${markdownTableCell(repoPath)}\``;
  const display = label || path.basename(repoPath);
  return `[${markdownTableCell(display)}](${href})`;
}

function markdownTableCell(value) {
  if (value == null) return '';
  return String(value).replaceAll('|', '\\|');
}

function formatSupportBadge(status) {
  if (status === CAPABILITY_FEATURE_STATUS.supported) return 'yes';
  if (status === CAPABILITY_FEATURE_STATUS.unsupported) return 'no';
  return CAPABILITY_FEATURE_STATUS.unknown;
}

function formatSamplingLabel(sampling) {
  if (!sampling || typeof sampling !== 'object') return '';
  const temp = asFiniteNumber(sampling.temperature);
  const topK = asFiniteNumber(sampling.topK);
  const topP = asFiniteNumber(sampling.topP);
  const isGreedy = temp === 0 && topK === 1 && (topP == null || topP === 1);
  if (isGreedy) return 'greedy (t=0)';
  const parts = [];
  if (temp != null) parts.push(`t=${temp}`);
  if (topK != null) parts.push(`k=${topK}`);
  if (topP != null) parts.push(`p=${topP}`);
  return parts.join(', ');
}

function asNonEmptyString(value) {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  return trimmed === '' ? null : trimmed;
}

function normalizeBackendLabel(backend, architecture) {
  const direct = asNonEmptyString(backend);
  if (direct) return direct.toLowerCase().startsWith('metal') ? 'metal' : direct;
  const arch = asNonEmptyString(architecture);
  if (arch && arch.toLowerCase().startsWith('metal')) return 'metal';
  return null;
}

function normalizeVendorLabel(vendor) {
  const value = asNonEmptyString(vendor);
  if (!value) return null;
  const lower = value.toLowerCase();
  if (lower === 'apple') return 'Apple';
  if (lower === 'unknown') return null;
  return value;
}

function normalizeBrowserLabel(browser) {
  const executable = asNonEmptyString(browser?.executable);
  const userAgent = asNonEmptyString(browser?.userAgent);
  const normalizeRawBrowserName = (value) => {
    const raw = asNonEmptyString(value);
    if (!raw) return null;
    const lower = raw.toLowerCase();
    if (lower.includes('headlesschrome') || lower.includes('chrome') || lower.includes('chromium')) {
      return 'chromium';
    }
    if (lower.includes('firefox')) return 'firefox';
    if (lower.includes('safari') && !lower.includes('chrome') && !lower.includes('chromium')) {
      return 'safari';
    }
    if (lower.includes('edg/') || lower.includes('edge')) return 'edge';
    return null;
  };
  return normalizeRawBrowserName(executable) || normalizeRawBrowserName(userAgent) || 'unknown';
}

function normalizeGpuDeviceLabel(gpu) {
  const canonicalizeAppleM3Label = (value) => {
    const normalized = asNonEmptyString(value);
    if (!normalized) return null;
    if (/^apple m3 pro$/i.test(normalized)) return 'Apple M3';
    if (/^m3 pro$/i.test(normalized)) return 'M3';
    return normalized;
  };
  const device = canonicalizeAppleM3Label(gpu?.device);
  const description = canonicalizeAppleM3Label(gpu?.description);
  const normalizedDevice = device && device.toLowerCase() !== 'unknown' && device !== '0x0000'
    ? device
    : null;
  const normalizedDescription = description && description.toLowerCase() !== 'unknown'
    ? description
    : null;
  return normalizedDevice || normalizedDescription || device || description || null;
}

function trimTrailingZeros(numberText) {
  if (typeof numberText !== 'string') return numberText;
  return numberText
    .replace(/\.0+$/, '')
    .replace(/(\.\d*?)0+$/, '$1');
}

function formatSizeLabel(sizeBytes) {
  const size = asFiniteNumber(sizeBytes);
  if (size == null || size <= 0) return null;
  const gib = 1024 ** 3;
  const mib = 1024 ** 2;
  if (size >= gib) {
    const value = size / gib;
    const digits = value >= 10 ? 1 : 2;
    return `${trimTrailingZeros(value.toFixed(digits))} GiB`;
  }
  if (size >= mib) {
    const value = size / mib;
    const digits = value >= 10 ? 1 : 2;
    return `${trimTrailingZeros(value.toFixed(digits))} MiB`;
  }
  return `${Math.round(size)} B`;
}

function formatMetricValue(value, unit) {
  const numeric = asFiniteNumber(value);
  if (numeric == null) return '';
  const absValue = Math.abs(numeric);
  const digits = absValue >= 100 ? 1 : (absValue >= 10 ? 2 : 3);
  const valueText = trimTrailingZeros(numeric.toFixed(digits));
  if (unit === '%') return `${valueText}%`;
  return unit ? `${valueText} ${unit}` : valueText;
}

function formatWorkloadModelLabel(compareResultSummary, modelCoverageById) {
  if (!compareResultSummary || typeof compareResultSummary !== 'object') return 'not captured';
  const modelId = asNonEmptyString(compareResultSummary.dopplerModelId);
  if (!modelId) return 'not captured';
  const modelCoverage = modelCoverageById.get(modelId) || null;
  const modelName = asNonEmptyString(modelCoverage?.catalogLabel) || modelId;
  const sizeLabel = formatSizeLabel(modelCoverage?.catalogSizeBytes);
  if (!sizeLabel) return modelName;
  return `${modelName} (${sizeLabel})`;
}

function formatRuntimeComboLabel(compareResultSummary) {
  const host = compareResultSummary?.environment?.host || {};
  const browser = compareResultSummary?.environment?.browser || {};
  const gpu = compareResultSummary?.environment?.gpu || {};
  const dopplerExecution = compareResultSummary?.dopplerExecution || {};
  const gpuVendor = normalizeVendorLabel(gpu.vendor);
  const gpuArchitecture = asNonEmptyString(gpu.architecture);
  const gpuDevice = normalizeGpuDeviceLabel(gpu);
  let gpuLabel = null;
  if (gpuDevice) {
    const vendorPrefix = gpuVendor && !gpuDevice.toLowerCase().includes(gpuVendor.toLowerCase())
      ? `${gpuVendor} `
      : '';
    gpuLabel = `${vendorPrefix}${gpuDevice}`;
  } else if (gpuVendor || gpuArchitecture) {
    gpuLabel = [gpuVendor, gpuArchitecture].filter((value) => value != null).join(' / ');
  }
  const backendLabel = normalizeBackendLabel(gpu.backend, gpu.architecture);
  const hostPlatform = asNonEmptyString(host.platform);
  const browserLabel = normalizeBrowserLabel(browser);
  const gpuValue = gpuLabel || 'unknown';
  const backendValue = backendLabel || 'unknown';
  const osValue = hostPlatform || 'unknown';
  const browserValue = browserLabel || 'unknown';
  const dopplerSurface = asNonEmptyString(compareResultSummary?.dopplerSurface);
  const cliExecutor = asNonEmptyString(dopplerExecution.cliExecutor);
  const commandSurface = asNonEmptyString(dopplerExecution.commandSurface);
  const dopplerParts = [
    dopplerSurface ? `doppler ${dopplerSurface}` : null,
    cliExecutor && dopplerSurface !== 'browser' && cliExecutor !== dopplerSurface ? `exec ${cliExecutor}` : null,
    commandSurface && commandSurface !== dopplerSurface ? `command ${commandSurface}` : null,
  ].filter(Boolean);
  const dopplerLabel = dopplerParts.length > 0 ? `; ${dopplerParts.join(' / ')}` : '';
  return `${gpuValue}; ${backendValue}; ${osValue}; ${browserValue}${dopplerLabel}`;
}

function formatCorrectnessLabel(compareResultSummary) {
  if (!compareResultSummary || typeof compareResultSummary !== 'object') return 'not captured';
  const correctness = compareResultSummary.correctness || {};
  if (correctness.exactMatch === true) return 'exact';
  if (correctness.normalizedMatch === true) return 'normalized';
  const status = asNonEmptyString(correctness.status);
  if (status) return status;
  return 'unknown';
}

function compareMetricLeader(metrics, metricId, higherBetter = true) {
  const metric = metrics && typeof metrics === 'object' && !Array.isArray(metrics)
    ? metrics[metricId]
    : null;
  const doppler = asFiniteNumber(metric?.doppler);
  const transformersjs = asFiniteNumber(metric?.transformersjs);
  if (doppler == null || transformersjs == null) return null;
  if (doppler === transformersjs) return 'tie';
  const dopplerLeads = higherBetter ? doppler > transformersjs : doppler < transformersjs;
  return dopplerLeads ? 'doppler' : 'transformersjs';
}

function formatMetricPair(doppler, transformersjs, unit) {
  const left = formatMetricValue(doppler, unit);
  const right = formatMetricValue(transformersjs, unit);
  if (!left && !right) return '';
  return `${left || 'n/a'} / ${right || 'n/a'}`;
}

function formatLeaderLabel(leader) {
  if (leader === 'doppler') return 'Doppler';
  if (leader === 'transformersjs') return 'TJS';
  if (leader === 'tie') return 'tie';
  return '';
}

function summarizeBottleneckLabel(bottleneck) {
  if (!bottleneck || typeof bottleneck !== 'object' || Array.isArray(bottleneck)) return null;
  const dominant = bottleneck.dominant || {};
  return asNonEmptyString(dominant.label) || asNonEmptyString(dominant.id);
}

function formatDopplerBottleneckLine(bottleneck) {
  if (!bottleneck || typeof bottleneck !== 'object' || Array.isArray(bottleneck)) return null;
  const dominant = bottleneck.dominant || {};
  const label = asNonEmptyString(dominant.label) || asNonEmptyString(dominant.id);
  if (!label) return null;
  const dominantMs = formatMetricValue(dominant.ms, 'ms');
  const share = asFiniteNumber(dominant.shareOfDecode);
  const shareText = share == null ? null : formatMetricValue(share * 100, '%');
  const commandRecordMs = asFiniteNumber(bottleneck.componentsMs?.commandRecordMs);
  const opCount = asFiniteNumber(bottleneck.recording?.opCount);
  const passCount = asFiniteNumber(bottleneck.recording?.passCount);
  const fragments = [`Doppler internal: ${label} ${dominantMs}`];
  if (shareText) {
    fragments.push(`${shareText} of decode`);
  }
  const dominantIsCommandRecord = /^command[ _-]?record/i.test(String(dominant.id || label || ''));
  if (commandRecordMs != null && !dominantIsCommandRecord) {
    fragments.push(`command recording ${formatMetricValue(commandRecordMs, 'ms')}`);
  }
  if (opCount != null && passCount != null) {
    fragments.push(`${formatMetricValue(opCount, null)} ops / ${formatMetricValue(passCount, null)} passes`);
  }
  return fragments.join('; ');
}

function summarizeClaimLaneGaps(lane, surfaces, laneRequiredBackendIds) {
  const capturedBackendIds = new Set(
    surfaces
      .map((surface) => asNonEmptyString(surface.backendId))
      .filter(Boolean)
  );
  const capturedWorkloadIds = new Set(
    surfaces
      .map((surface) => asNonEmptyString(surface.workloadId))
      .filter(Boolean)
  );
  const capturedDecodeProfileIds = new Set();
  for (const surface of surfaces) {
    const sectionIds = Array.isArray(surface.computeSectionIds) ? surface.computeSectionIds : [];
    for (const sectionId of sectionIds) {
      const normalized = asNonEmptyString(sectionId);
      if (normalized) capturedDecodeProfileIds.add(normalized);
    }
    const decodeProfile = asNonEmptyString(surface.decodeProfile);
    if (decodeProfile) capturedDecodeProfileIds.add(decodeProfile);
  }
  const requiredBackendIds = Array.from(laneRequiredBackendIds).sort();
  const requiredWorkloadIds = Array.isArray(lane?.run?.workloads) ? lane.run.workloads : [];
  const requiredDecodeProfileIds = Array.isArray(lane?.run?.decodeProfiles) ? lane.run.decodeProfiles : [];
  const capturedSurfaceWorkloads = new Set(
    surfaces
      .map((surface) => {
        const backendId = asNonEmptyString(surface.backendId);
        const workloadId = asNonEmptyString(surface.workloadId);
        return backendId && workloadId ? `${backendId}:${workloadId}` : null;
      })
      .filter(Boolean)
  );
  const requiredSurfaceWorkloads = [];
  for (const backendId of requiredBackendIds) {
    for (const workloadId of requiredWorkloadIds) {
      requiredSurfaceWorkloads.push(`${backendId}:${workloadId}`);
    }
  }
  const missingBackendIds = requiredBackendIds.filter((backendId) => !capturedBackendIds.has(backendId));
  const missingWorkloadIds = requiredWorkloadIds.filter((workloadId) => !capturedWorkloadIds.has(workloadId));
  const missingDecodeProfileIds = requiredDecodeProfileIds.filter(
    (decodeProfileId) => !capturedDecodeProfileIds.has(decodeProfileId)
  );
  const missingSurfaceWorkloads = requiredSurfaceWorkloads.filter((entry) => !capturedSurfaceWorkloads.has(entry));
  const claimReady = lane?.status === 'promoted'
    && missingBackendIds.length === 0
    && missingWorkloadIds.length === 0
    && missingDecodeProfileIds.length === 0
    && missingSurfaceWorkloads.length === 0
    && surfaces.length > 0;
  return {
    claimReady,
    missingBackendIds,
    missingWorkloadIds,
    missingDecodeProfileIds,
    missingSurfaceWorkloads,
  };
}

function formatClaimGapList(values) {
  if (!Array.isArray(values)) return '';
  if (values.length <= 6) return values.join(', ');
  return `${values.slice(0, 6).join(', ')} +${values.length - 6} more`;
}

function formatClaimLaneGateGaps(lane) {
  if (lane?.claimReady === true) return 'ready';
  const parts = [];
  if (asNonEmptyString(lane?.status) && lane.status !== 'promoted') {
    parts.push(`status ${lane.status}`);
  }
  if (Array.isArray(lane?.missingBackendIds) && lane.missingBackendIds.length > 0) {
    parts.push(`missing backends ${lane.missingBackendIds.join(', ')}`);
  }
  if (Array.isArray(lane?.missingWorkloadIds) && lane.missingWorkloadIds.length > 0) {
    parts.push(`missing workloads ${lane.missingWorkloadIds.join(', ')}`);
  }
  if (Array.isArray(lane?.missingDecodeProfileIds) && lane.missingDecodeProfileIds.length > 0) {
    parts.push(`missing decode profiles ${lane.missingDecodeProfileIds.join(', ')}`);
  }
  if (Array.isArray(lane?.missingSurfaceWorkloads) && lane.missingSurfaceWorkloads.length > 0) {
    parts.push(`missing backend/workload ${formatClaimGapList(lane.missingSurfaceWorkloads)}`);
  }
  return parts.length > 0 ? parts.join('; ') : 'not promoted';
}

async function buildLocalClaimLaneSummaries(localInferenceClaimMatrix, compareMetricBundle) {
  const requiredBackendById = new Map(
    (Array.isArray(localInferenceClaimMatrix?.sharedRunContract?.requiredRuntimeBackends)
      ? localInferenceClaimMatrix.sharedRunContract.requiredRuntimeBackends
      : [])
      .map((entry) => [entry.id, entry])
  );
  const metricIds = Array.isArray(compareMetricBundle?.metricIds) ? compareMetricBundle.metricIds : null;
  const metricDefinitions = Array.isArray(compareMetricBundle?.metrics) ? compareMetricBundle.metrics : null;
  const out = [];
  const lanes = Array.isArray(localInferenceClaimMatrix?.lanes) ? localInferenceClaimMatrix.lanes : [];
  for (const lane of lanes) {
    const laneLabel = `local claim lane ${lane?.id || 'unknown'}`;
    const evidence = lane?.evidence && typeof lane.evidence === 'object' && !Array.isArray(lane.evidence)
      ? lane.evidence
      : {};
    const surfaceEntries = Array.isArray(evidence.workloadCompareResults) && evidence.workloadCompareResults.length > 0
      ? evidence.workloadCompareResults
      : Array.isArray(evidence.surfaceCompareResults) && evidence.surfaceCompareResults.length > 0
        ? evidence.surfaceCompareResults
      : (evidence.compareResult
          ? [{
              backendId: null,
              compareResult: evidence.compareResult,
              summarySvg: evidence.summarySvg ?? null,
            }]
          : []);
    const surfaces = [];
    for (const [surfaceIndex, surfaceEvidence] of surfaceEntries.entries()) {
      const evidenceCollection = surfaceEvidence.workloadId ? 'workloadCompareResults' : 'surfaceCompareResults';
      const evidenceLabel = `${laneLabel}.evidence.${evidenceCollection}[${surfaceIndex}]`;
      const compareResultPath = await assertRepoPathExists(
        surfaceEvidence.compareResult,
        `${evidenceLabel}.compareResult`
      );
      if (surfaceEvidence.summarySvg != null) {
        await assertRepoPathExists(surfaceEvidence.summarySvg, `${evidenceLabel}.summarySvg`);
      }
      const summary = await maybeLoadCompareResultSummary(
        compareResultPath,
        metricIds,
        metricDefinitions
      );
      const backendId = asNonEmptyString(surfaceEvidence.backendId);
      const requiredBackend = backendId ? requiredBackendById.get(backendId) || null : null;
      const decodeMetric = summary?.metrics?.decodeTokensPerSec || {};
      const promptMetric = summary?.metrics?.promptTokensPerSecToFirstToken || {};
      surfaces.push({
        backendId,
        surface: asNonEmptyString(requiredBackend?.surface) || asNonEmptyString(summary?.dopplerSurface),
        compareResult: summary?.path || toPosixRelative(compareResultPath),
        summarySvg: asNonEmptyString(surfaceEvidence.summarySvg),
        workloadId: asNonEmptyString(surfaceEvidence.workloadId) || asNonEmptyString(summary?.workloadId),
        decodeProfile: asNonEmptyString(summary?.decodeProfile),
        correctness: formatCorrectnessLabel(summary),
        dopplerDecodeTokensPerSec: asFiniteNumber(decodeMetric.doppler),
        transformersjsDecodeTokensPerSec: asFiniteNumber(decodeMetric.transformersjs),
        decodeLeader: compareMetricLeader(summary?.metrics, 'decodeTokensPerSec', true),
        dopplerPromptTokensPerSecToFirstToken: asFiniteNumber(promptMetric.doppler),
        transformersjsPromptTokensPerSecToFirstToken: asFiniteNumber(promptMetric.transformersjs),
        promptLeader: compareMetricLeader(summary?.metrics, 'promptTokensPerSecToFirstToken', true),
        bottleneck: summarizeBottleneckLabel(summary?.dopplerBottleneck),
        bottleneckClass: asNonEmptyString(summary?.dopplerBottleneck?.bottleneckClass),
        computeSectionIds: Array.isArray(summary?.computeSectionIds) ? summary.computeSectionIds : [],
      });
    }
    surfaces.sort((left, right) => {
      const leftKey = asNonEmptyString(left.backendId) || asNonEmptyString(left.surface) || '';
      const rightKey = asNonEmptyString(right.backendId) || asNonEmptyString(right.surface) || '';
      const byBackend = leftKey.localeCompare(rightKey);
      if (byBackend !== 0) return byBackend;
      const leftWorkload = asNonEmptyString(left.workloadId) || '';
      const rightWorkload = asNonEmptyString(right.workloadId) || '';
      return leftWorkload.localeCompare(rightWorkload);
    });
    const laneRequiredBackendIds = new Set(getLaneRequiredBackendIds(lane, localInferenceClaimMatrix));
    const claimGaps = summarizeClaimLaneGaps(lane, surfaces, laneRequiredBackendIds);
    out.push({
      laneId: lane.id,
      status: asNonEmptyString(lane.status) || 'unknown',
      claimReady: claimGaps.claimReady,
      statusReason: asNonEmptyString(lane.statusReason),
      missingBackendIds: claimGaps.missingBackendIds,
      missingWorkloadIds: claimGaps.missingWorkloadIds,
      missingDecodeProfileIds: claimGaps.missingDecodeProfileIds,
      missingSurfaceWorkloads: claimGaps.missingSurfaceWorkloads,
      dopplerModelId: asNonEmptyString(lane?.model?.dopplerModelId) || null,
      modelLabel: asNonEmptyString(lane?.model?.label) || null,
      surfaces,
    });
  }
  return out.sort((left, right) => left.laneId.localeCompare(right.laneId));
}

function renderReleaseMatrixMarkdown(matrix, options = {}) {
  const markdownPath = typeof options.markdownPath === 'string'
    ? path.resolve(options.markdownPath)
    : null;
  const compareResults = Array.isArray(options.compareResults)
    ? [...options.compareResults].sort(compareResultSortDescending)
    : [];
  const compareResultsForDisplay = selectLatestCompareResultsByType(compareResults);
  const modelCoverageById = new Map();
  for (const row of Array.isArray(matrix.modelCoverage) ? matrix.modelCoverage : []) {
    const modelId = asNonEmptyString(row?.dopplerModelId);
    if (!modelId) continue;
    modelCoverageById.set(modelId, row);
  }
  const compareResultsByWorkload = new Map();
  for (const result of compareResultsForDisplay) {
    if (typeof result?.workloadId !== 'string' || result.workloadId.trim() === '') continue;
    const bucket = compareResultsByWorkload.get(result.workloadId) || [];
    bucket.push(result);
    compareResultsByWorkload.set(result.workloadId, bucket);
  }

  const lines = [];
  lines.push('# Release Matrix');
  lines.push('');
  lines.push(`Generated: ${matrix.generatedAt}`);
  const releaseChannel = matrix?.release?.channel || 'n/a';
  const releaseVersion = matrix?.release?.version || 'n/a';
  const releaseCommit = matrix?.release?.commitSha || 'n/a';
  const releaseDirty = matrix?.release?.dirty === true
    ? 'yes'
    : (matrix?.release?.dirty === false ? 'no' : 'n/a');
  lines.push(`Release: channel=${releaseChannel}, version=${releaseVersion}, commit=${releaseCommit}, dirty=${releaseDirty}`);
  lines.push('');
  lines.push('## Engine Matrix');
  lines.push('');
  lines.push('| Target | Status | Browser WebGPU | Node WebGPU | Headless Harness | Cache Mode Control | Bench Features | Profile Features |');
  lines.push('|---|---|---|---|---|---|---:|---:|');
  for (const target of matrix.targets) {
    lines.push(
      `| ${markdownTableCell(target.name)} (\`${target.id}\`) | ${markdownTableCell(target.status)} | `
      + `${formatSupportBadge(target.platformSupport.browserWebgpu)} | `
      + `${formatSupportBadge(target.platformSupport.nodeWebgpu)} | `
      + `${formatSupportBadge(target.platformSupport.headlessBrowserHarness)} | `
      + `${formatSupportBadge(target.platformSupport.cacheModeControl)} | `
      + `${target.bench.enabledFeatures}/${target.bench.totalFeatures} | `
      + `${target.profile.enabledFeatures}/${target.profile.totalFeatures} |`
    );
  }
  lines.push('');
  lines.push('## Model Coverage');
  lines.push('');
  lines.push('| Doppler Model | In Catalog | Catalog Modes | TJS Mapping | Surface | Source | Compare Lane | Notes |');
  lines.push('|---|---|---|---|---|---|---|---|');
  for (const row of matrix.modelCoverage) {
    lines.push(
      `| \`${markdownTableCell(row.dopplerModelId)}\` | ${row.inCatalog ? 'yes' : 'no'} | `
      + `${markdownTableCell(row.catalogModes.join(', '))} | `
      + `${row.defaultTjsModelId ? `\`${markdownTableCell(row.defaultTjsModelId)}\`` : ''} | `
      + `${markdownTableCell(row.defaultDopplerSurface || '')} | `
      + `${markdownTableCell(row.dopplerSource || '')} | `
      + `${markdownTableCell(row.compareLane || '')} | `
      + `${markdownTableCell(row.compareLaneReason || '')} |`
    );
  }
  lines.push('');
  lines.push('## Workloads');
  lines.push('');
  lines.push('| Workload ID | Model | Prefill | Decode | Sampling | Correctness | Runtime (GPU/Backend/OS/Browser) | Date |');
  lines.push('|---|---|---:|---:|---|---|---|---|');
  for (const workload of matrix.workloads) {
    const workloadRuns = compareResultsByWorkload.get(workload.id) || [];
    const sortedRuns = [...workloadRuns].sort((left, right) => {
      const leftKey = compareResultModelTypeKey(left);
      const rightKey = compareResultModelTypeKey(right);
      return leftKey.localeCompare(rightKey);
    });
    const runsToRender = sortedRuns.length > 0 ? sortedRuns : [null];
    for (const selectedRun of runsToRender) {
      const workloadIdCell = selectedRun
        ? formatRepoPathLink(selectedRun.path, markdownPath, `\`${workload.id}\``)
        : `\`${markdownTableCell(workload.id)}\``;
      const modelCell = markdownTableCell(formatWorkloadModelLabel(selectedRun, modelCoverageById));
      const runtimeComboCell = selectedRun
        ? markdownTableCell(formatRuntimeComboLabel(selectedRun))
        : 'not captured';
      const correctnessCell = selectedRun
        ? markdownTableCell(formatCorrectnessLabel(selectedRun))
        : 'not captured';
      const dateCell = selectedRun && typeof selectedRun.timestamp === 'string' && selectedRun.timestamp.length >= 10
        ? markdownTableCell(selectedRun.timestamp.slice(0, 10))
        : (selectedRun ? 'captured' : 'not captured');
      lines.push(
        `| ${workloadIdCell} | ${modelCell} | `
        + `${workload.prefillTokens ?? ''} | ${workload.decodeTokens ?? ''} | `
        + `${markdownTableCell(formatSamplingLabel(workload.sampling))} | ${correctnessCell} | `
        + `${runtimeComboCell} | ${dateCell} |`
      );
    }
  }
  lines.push('');
  lines.push('## Local Claim Lanes');
  lines.push('');
  lines.push('| Lane | Status | Gate gaps | Backend | Surface | Workload | Decode tok/s (Doppler/TJS) | Prompt tok/s (Doppler/TJS) | Leaders | Bottleneck | Evidence |');
  lines.push('|---|---|---|---|---|---|---:|---:|---|---|---|');
  const localClaimLanes = Array.isArray(matrix.localClaimLanes) ? matrix.localClaimLanes : [];
  if (localClaimLanes.length > 0) {
    for (const lane of localClaimLanes) {
      const laneLabel = lane.modelLabel
        ? `${lane.laneId} (${lane.modelLabel})`
        : lane.laneId;
      const gateGapCell = formatClaimLaneGateGaps(lane);
      const surfaces = Array.isArray(lane.surfaces) && lane.surfaces.length > 0
        ? lane.surfaces
        : [null];
      for (const surface of surfaces) {
        const evidenceCell = surface?.compareResult
          ? [
              formatRepoPathLink(surface.compareResult, markdownPath, 'compare'),
              surface.summarySvg ? formatRepoPathLink(surface.summarySvg, markdownPath, 'svg') : null,
            ].filter(Boolean).join(' / ')
          : 'missing';
        const bottleneckCell = surface?.bottleneck
          ? [
              surface.bottleneck,
              surface.bottleneckClass ? `(${surface.bottleneckClass})` : null,
            ].filter(Boolean).join(' ')
          : '';
        const leaderCell = [
          surface?.decodeLeader ? `decode ${formatLeaderLabel(surface.decodeLeader)}` : null,
          surface?.promptLeader ? `prompt ${formatLeaderLabel(surface.promptLeader)}` : null,
        ].filter(Boolean).join('; ');
        lines.push(
          `| \`${markdownTableCell(laneLabel)}\` | ${markdownTableCell(lane.status)} | `
          + `${markdownTableCell(gateGapCell)} | `
          + `${markdownTableCell(surface?.backendId || 'not captured')} | `
          + `${markdownTableCell(surface?.surface || 'not captured')} | `
          + `${markdownTableCell(surface?.workloadId || 'not captured')} | `
          + `${markdownTableCell(formatMetricPair(surface?.dopplerDecodeTokensPerSec, surface?.transformersjsDecodeTokensPerSec, 'tok/s'))} | `
          + `${markdownTableCell(formatMetricPair(surface?.dopplerPromptTokensPerSecToFirstToken, surface?.transformersjsPromptTokensPerSecToFirstToken, 'tok/s'))} | `
          + `${markdownTableCell(leaderCell)} | ${markdownTableCell(bottleneckCell)} | ${evidenceCell} |`
        );
      }
    }
  } else {
    lines.push('| not captured | not captured | not captured | not captured | not captured | not captured |  |  |  |  | missing |');
  }
  lines.push('');
  lines.push('## Latest Bottlenecks');
  lines.push('');
  const latestCompareResult = matrix.evidence?.latestCompareResult || null;
  const latestBottlenecks = Array.isArray(latestCompareResult?.bottlenecks)
    ? latestCompareResult.bottlenecks.slice(0, 6)
    : [];
  if (latestBottlenecks.length > 0) {
    const latestPath = formatRepoPathLink(latestCompareResult.path, markdownPath);
    lines.push(`Source: ${latestPath}`);
    lines.push('');
    const dopplerBottleneckLine = formatDopplerBottleneckLine(latestCompareResult.dopplerBottleneck);
    if (dopplerBottleneckLine) {
      lines.push(dopplerBottleneckLine);
      lines.push('');
    }
    lines.push('| Metric | Leader | Gap | Doppler | Transformers.js |');
    lines.push('|---|---|---:|---:|---:|');
    for (const bottleneck of latestBottlenecks) {
      lines.push(
        `| ${markdownTableCell(bottleneck.label || bottleneck.metricId)} | `
        + `${markdownTableCell(bottleneck.leader)} | `
        + `${formatMetricValue(bottleneck.gapPercent, '%')} | `
        + `${markdownTableCell(formatMetricValue(bottleneck.doppler, bottleneck.unit))} | `
        + `${markdownTableCell(formatMetricValue(bottleneck.transformersjs, bottleneck.unit))} |`
      );
    }
  } else {
    lines.push('- no TJS-leading metrics in the latest selected compare result');
  }
  lines.push('');
  lines.push('## Charts');
  lines.push('');
  if (matrix.evidence.committedCharts.length > 0) {
    for (const chartPath of matrix.evidence.committedCharts) {
      lines.push(`- ${formatRepoPathLink(chartPath, markdownPath)}`);
    }
  } else {
    lines.push('- none');
  }
  return `${lines.join('\n')}\n`;
}

async function doMatrix(flags, timestamp = null) {
  const { registry, workloads } = await loadRegistryBundle();
  const capabilities = await loadCapabilitiesBundle(registry);
  const compareConfig = await loadCompareConfigBundle();
  const compareMetricBundle = await loadCompareMetricBundle();
  const catalog = await loadModelCatalogBundle();
  const benchmarkPolicy = await readJson(BENCHMARK_POLICY_PATH);
  const localInferenceClaimMatrix = await loadLocalInferenceClaimMatrixBundle({
    registry,
    workloads,
    compareConfig,
    compareMetricBundle,
    benchmarkPolicy,
    catalog,
    capabilities,
  });

  const compareResultFlag = flags['compare-result'] ?? null;
  const includeLocalResults = parseBooleanFlag(
    flags['include-local-results'],
    false,
    '--include-local-results'
  );
  const strictCompareArtifacts = parseBooleanFlag(
    flags['strict-compare-artifacts'],
    false,
    '--strict-compare-artifacts'
  );
  const {
    compareResults,
    latestCompareResult,
    droppedCompareArtifacts,
  } = await loadCompareResultSummaries({
    compareResultFlag,
    compareMetricIds: compareMetricBundle.metricIds,
    compareMetricDefinitions: compareMetricBundle.metrics,
    includeLocalResults,
    strictCompareArtifacts,
  });
  if (droppedCompareArtifacts.length > 0) {
    for (const dropped of droppedCompareArtifacts) {
      console.error(
        `[vendor-bench] warning: skipped compare artifact "${dropped.path}" `
        + `(${dropped.origin}): ${dropped.error}`
      );
    }
  }
  const committedCharts = await listCommittedCharts();
  const releaseMetadata = resolveGitReleaseMetadata();
  const packageJson = await readJson(PACKAGE_JSON_PATH);
  const releaseVersion = asNonEmptyStringValue(packageJson?.version);
  const releaseClaimableModelIds = new Set(
    [...catalog.byModelId.values()]
      .filter((entry) => entry.releaseClaimable === true)
      .map((entry) => entry.modelId)
  );

  const benchFeatureIds = capabilities.featureCatalog.bench.map((entry) => entry.id);
  const profileFeatureIds = capabilities.featureCatalog.profile.map((entry) => entry.id);

  const targets = registry.products.map((product) => {
    const capability = resolveCapabilityTarget(capabilities, product.id);
    const benchFeatures = normalizeCapabilityFeatureMap(capability?.bench?.features, benchFeatureIds);
    const profileFeatures = normalizeCapabilityFeatureMap(capability?.profile?.features, profileFeatureIds);
    return {
      id: product.id,
      name: product.name,
      status: product.status,
      category: product.category,
      website: product.website,
      platformSupport: {
        browserWebgpu: supportStatus(benchFeatures.browser_webgpu),
        nodeWebgpu: supportStatus(benchFeatures.node_webgpu),
        headlessBrowserHarness: supportStatus(benchFeatures.headless_browser_harness),
        cacheModeControl: supportStatus(benchFeatures.cache_mode_control),
      },
      bench: {
        enabledFeatures: countEnabledFeatures(benchFeatures),
        totalFeatures: benchFeatureIds.length,
        features: benchFeatures,
      },
      profile: {
        enabledFeatures: countEnabledFeatures(profileFeatures),
        totalFeatures: profileFeatureIds.length,
        features: profileFeatures,
      },
      evidence: Array.isArray(capability?.evidence) ? capability.evidence : [],
    };
  });

  const coveredModelIds = new Set();
  const modelCoverage = [];
  for (const profile of compareConfig.modelProfiles) {
    const dopplerModelId = profile.dopplerModelId;
    if (!releaseClaimableModelIds.has(dopplerModelId)) continue;
    coveredModelIds.add(dopplerModelId);
    const catalogEntry = catalog.byModelId.get(dopplerModelId) || null;
    modelCoverage.push({
      dopplerModelId,
      dopplerModelBaseDir: profile.modelBaseDir || null,
      dopplerSource: profile.defaultDopplerSource || 'local',
      defaultTjsModelId: profile.defaultTjsModelId || null,
      defaultDopplerSurface: profile.defaultDopplerSurface || 'auto',
      compareLane: profile.compareLane || 'performance_comparable',
      compareLaneReason: profile.compareLaneReason || null,
      inCatalog: Boolean(catalogEntry),
      catalogLabel: catalogEntry?.label || null,
      catalogModes: Array.isArray(catalogEntry?.modes) ? catalogEntry.modes : [],
      catalogBaseUrl: catalogEntry?.baseUrl || null,
      catalogSizeBytes: catalogEntry?.sizeBytes ?? null,
    });
  }
  for (const catalogEntry of catalog.byModelId.values()) {
    if (!catalogEntry.releaseClaimable || coveredModelIds.has(catalogEntry.modelId)) continue;
    modelCoverage.push({
      dopplerModelId: catalogEntry.modelId,
      dopplerModelBaseDir: null,
      dopplerSource: null,
      defaultTjsModelId: null,
      defaultDopplerSurface: 'auto',
      compareLane: null,
      compareLaneReason: null,
      inCatalog: true,
      catalogLabel: catalogEntry.label,
      catalogModes: catalogEntry.modes,
      catalogBaseUrl: catalogEntry.baseUrl,
      catalogSizeBytes: catalogEntry.sizeBytes,
    });
  }
  modelCoverage.sort((a, b) => a.dopplerModelId.localeCompare(b.dopplerModelId));
  const coveredModelIdSet = new Set(modelCoverage.map((entry) => entry.dopplerModelId));
  const releaseCompareResults = compareResults.filter((entry) => {
    const modelId = asNonEmptyString(entry?.dopplerModelId);
    const comparableLane = entry?.compareLane == null || entry.compareLane === 'performance_comparable';
    const fairnessAllowsRelease = entry?.fairness?.releaseClaimable === true;
    return modelId != null
      && coveredModelIdSet.has(modelId)
      && comparableLane
      && fairnessAllowsRelease
      && entry?.pairedComparable !== false;
  });
  const latestReleaseCompareResult = selectLatestCompareResult(releaseCompareResults);

  const normalizedWorkloads = workloads.workloads.map((workload) => ({
    id: workload.id,
    name: workload.name || null,
    prefillTokens: asFiniteNumber(workload.prefillTokens),
    decodeTokens: asFiniteNumber(workload.decodeTokens),
    sampling: workload?.sampling && typeof workload.sampling === 'object'
      ? {
          temperature: asFiniteNumber(workload.sampling.temperature),
          topK: asFiniteNumber(workload.sampling.topK),
          topP: asFiniteNumber(workload.sampling.topP),
        }
      : null,
  }));

  const sourceHashes = {};
  const sourceFiles = [
    ['registry', REGISTRY_PATH],
    ['workloads', WORKLOADS_PATH],
    ['capabilities', CAPABILITIES_PATH],
    ['compareConfig', COMPARE_CONFIG_PATH],
    ['compareMetricContract', COMPARE_METRIC_CONTRACT_PATH],
    ['benchmarkPolicy', BENCHMARK_POLICY_PATH],
    ['localInferenceClaimMatrix', LOCAL_INFERENCE_CLAIM_MATRIX_PATH],
    ['modelCatalog', MODEL_CATALOG_PATH],
  ];
  for (const product of registry.products) {
    sourceFiles.push([`harness:${product.id}`, path.join(REGISTRY_DIR, product.harness)]);
  }
  for (const [key, filePath] of sourceFiles) {
    const payload = await readJson(filePath);
    const hashInfo = await hashJsonBytes(payload);
    sourceHashes[key] = {
      path: toPosixRelative(filePath),
      sha256: hashInfo.sha256,
      bytes: hashInfo.bytes,
      updated: typeof payload?.updated === 'string'
        ? payload.updated
        : (typeof payload?.updatedAt === 'string' ? payload.updatedAt : null),
    };
  }

  const matrix = {
    schemaVersion: 1,
    generatedAt: toIsoTimestamp(timestamp),
    release: {
      channel: 'main-snapshot',
      version: releaseVersion,
      commitSha: releaseMetadata.commitSha,
      dirty: releaseMetadata.dirty,
    },
    sources: {
      ...sourceHashes,
    },
    summary: {
      targetCount: targets.length,
      modelCoverageCount: modelCoverage.length,
      compareProfileCount: compareConfig.modelProfiles.length,
      catalogModelCount: catalog.byModelId.size,
      workloadCount: normalizedWorkloads.length,
      committedChartCount: committedCharts.length,
      hasLatestCompareResult: latestReleaseCompareResult != null,
    },
    targets,
    modelCoverage,
    workloads: normalizedWorkloads,
    localClaimLanes: await buildLocalClaimLaneSummaries(
      localInferenceClaimMatrix,
      compareMetricBundle
    ),
    evidence: {
      committedCharts,
      latestCompareResult: latestReleaseCompareResult,
    },
  };

  await assertMatchesSchema(matrix, RELEASE_MATRIX_SCHEMA_PATH, 'release-matrix');
  const outputPath = flags.output
    ? path.resolve(flags.output)
    : DEFAULT_RELEASE_MATRIX_OUTPUT_PATH;
  const markdownPath = flags['markdown-output']
    ? path.resolve(flags['markdown-output'])
    : DEFAULT_RELEASE_MATRIX_MARKDOWN_PATH;
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, `${JSON.stringify(matrix, null, 2)}\n`, 'utf8');
  await fs.mkdir(path.dirname(markdownPath), { recursive: true });
  await fs.writeFile(markdownPath, renderReleaseMatrixMarkdown(matrix, {
    markdownPath,
    compareResults: releaseCompareResults,
  }), 'utf8');
  console.log(outputPath);
  console.log(markdownPath);
}

async function runCommandCaptureJson(commandParts, options = {}) {
  if (!Array.isArray(commandParts) || commandParts.length === 0) {
    throw new Error('No command provided for run mode');
  }
  const [command, ...args] = commandParts;
  const timeoutMs = parsePositiveInteger(options.timeoutMs, DEFAULT_COMMAND_TIMEOUT_MS, '--timeout-ms');
  const telemetry = createResourceTelemetrySampler(options.resourceTelemetry);
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: ROOT_DIR,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: process.env,
    });
    telemetry.start(child.pid);

    const timeoutLabel = `${command} ${args.join(' ')}`.trim();
    let killedByTimeout = false;
    let timeoutHandle;
    if (Number.isFinite(timeoutMs) && timeoutMs > 0) {
      timeoutHandle = setTimeout(() => {
        killedByTimeout = true;
        child.kill('SIGKILL');
      }, timeoutMs);
    }

    const stdoutChunks = [];
    const stderrChunks = [];
    child.stdout.on('data', (chunk) => stdoutChunks.push(chunk));
    child.stderr.on('data', (chunk) => stderrChunks.push(chunk));
    child.on('error', (error) => {
      if (timeoutHandle) clearTimeout(timeoutHandle);
      telemetry.stop();
      reject(error);
    });
    child.on('close', (code) => {
      if (timeoutHandle) clearTimeout(timeoutHandle);
      const resourceTelemetry = telemetry.stop();
      if (killedByTimeout) {
        reject(new Error(`Command timed out after ${timeoutMs}ms: ${timeoutLabel}`));
        return;
      }
      const stdout = Buffer.concat(stdoutChunks).toString('utf8').trim();
      const stderr = Buffer.concat(stderrChunks).toString('utf8').trim();
      if (code !== 0) {
        reject(new Error(`Command failed (${code}): ${stderr || commandParts.join(' ')}`));
        return;
      }
      try {
        const payload = parseJsonFromStdout(stdout, timeoutLabel);
        resolve({ payload, resourceTelemetry });
      } catch (error) {
        reject(new Error(`Command stdout was not valid JSON: ${error.message}`));
      }
    });
  });
}

async function assertReleaseMatrixSourceHashes(matrix) {
  const sourceRows = Object.entries(matrix?.sources || {});
  const staleRows = [];
  for (const [sourceId, sourceInfo] of sourceRows) {
    const sourcePath = asNonEmptyStringValue(sourceInfo?.path);
    if (!sourcePath) continue;
    const absolutePath = path.resolve(ROOT_DIR, sourcePath);
    if (!(await fileExists(absolutePath))) {
      staleRows.push(`${sourceId}: source file missing (${sourcePath})`);
      continue;
    }

    let payload;
    try {
      payload = await readJson(absolutePath);
    } catch (error) {
      staleRows.push(`${sourceId}: could not parse source JSON (${sourcePath}): ${error.message}`);
      continue;
    }

    const currentHash = await hashJsonBytes(payload);
    if (currentHash.sha256 !== sourceInfo.sha256 || currentHash.bytes !== sourceInfo.bytes) {
      staleRows.push(
        `${sourceId}: hash mismatch for ${sourcePath} `
        + `(expected sha256=${sourceInfo.sha256}, bytes=${sourceInfo.bytes}; `
        + `actual sha256=${currentHash.sha256}, bytes=${currentHash.bytes})`
      );
    }
  }
  if (staleRows.length > 0) {
    throw new Error(
      'release-matrix.json sources are stale:\n'
      + staleRows.map((row) => `- ${row}`).join('\n')
      + '\nRegenerate with: node tools/vendor-bench.js matrix'
    );
  }
}

async function doValidate() {
  const { registry, workloads } = await loadRegistryBundle();
  const capabilities = await loadCapabilitiesBundle(registry);
  const compareConfig = await loadCompareConfigBundle();
  const compareMetricBundle = await loadCompareMetricBundle();
  const catalog = await loadModelCatalogBundle();
  const benchmarkPolicy = await readJson(BENCHMARK_POLICY_PATH);
  await loadLocalInferenceClaimMatrixBundle({
    registry,
    workloads,
    compareConfig,
    compareMetricBundle,
    benchmarkPolicy,
    catalog,
    capabilities,
  });
  for (const product of registry.products) {
    const harnessPath = path.join(REGISTRY_DIR, product.harness);
    const exists = await fileExists(harnessPath);
    if (!exists) {
      throw new Error(`Missing harness file for ${product.id}: ${product.harness}`);
    }
    const harness = await readJson(harnessPath);
    await assertMatchesSchema(harness, HARNESS_SCHEMA_PATH, `harness ${harnessPath}`);
    assertHarnessShape(harness, product.id);
  }
  const releaseMatrixExists = await fileExists(DEFAULT_RELEASE_MATRIX_OUTPUT_PATH);
  if (releaseMatrixExists) {
    const matrix = await readJson(DEFAULT_RELEASE_MATRIX_OUTPUT_PATH);
    await assertMatchesSchema(matrix, RELEASE_MATRIX_SCHEMA_PATH, 'release-matrix.json');
    await assertReleaseMatrixSourceHashes(matrix);
  }
  console.log(`OK registry: ${registry.products.length} products, ${workloads.workloads.length} workloads, capabilities`);
}

async function doCapabilities(flags) {
  const { registry } = await loadRegistryBundle();
  const capabilities = await loadCapabilitiesBundle(registry);
  const targetId = flags.target ?? null;

  if (targetId) {
    const target = resolveCapabilityTarget(capabilities, targetId);
    console.log(JSON.stringify(target, null, 2));
    return;
  }

  console.log('id\tbench\tprofile\tname');
  for (const target of capabilities.targets) {
    const benchCount = countEnabledFeatures(target.bench?.features);
    const profileCount = countEnabledFeatures(target.profile?.features);
    console.log(`${target.id}\t${benchCount}\t${profileCount}\t${target.name}`);
  }
}

async function doGap(flags) {
  const baseId = flags.base;
  const targetId = flags.target;
  if (!baseId) {
    throw new Error('gap requires --base <id>');
  }
  if (!targetId) {
    throw new Error('gap requires --target <id>');
  }

  const { registry } = await loadRegistryBundle();
  const capabilities = await loadCapabilitiesBundle(registry);
  const base = resolveCapabilityTarget(capabilities, baseId);
  const target = resolveCapabilityTarget(capabilities, targetId);

  const benchMissing = listMissingFeatures(base.bench?.features || {}, target.bench?.features || {});
  const profileMissing = listMissingFeatures(base.profile?.features || {}, target.profile?.features || {});
  const benchExtra = listExtraFeatures(base.bench?.features || {}, target.bench?.features || {});
  const profileExtra = listExtraFeatures(base.profile?.features || {}, target.profile?.features || {});

  console.log(JSON.stringify({
    base: { id: base.id, name: base.name },
    target: { id: target.id, name: target.name },
    benchMissing,
    profileMissing,
    benchExtra,
    profileExtra,
  }, null, 2));
}

async function doShow(flags) {
  const targetId = flags.target;
  if (!targetId) {
    throw new Error('show requires --target <id>');
  }
  const { registry } = await loadRegistryBundle();
  const { product, harness } = await loadTargetBundle(targetId, registry);
  console.log(JSON.stringify({ product, harness }, null, 2));
}

async function doImport(flags, timestamp = null) {
  const targetId = flags.target;
  const inputPath = flags.input;
  if (!targetId) {
    throw new Error('import requires --target <id>');
  }
  if (!inputPath) {
    throw new Error('import requires --input <raw.json>');
  }

  const { registry, workloads } = await loadRegistryBundle();
  const { product, harness } = await loadTargetBundle(targetId, registry);
  const rawResult = await readJson(path.resolve(inputPath));
  const rawInfo = await hashJsonBytes(rawResult);
  const workloadsById = new Map(workloads.workloads.map((item) => [item.id, item]));

  const record = await normalizeRecord({
    product,
    harness,
    rawResult,
    workloadsById,
    workloadId: flags.workload ?? null,
    modelId: flags.model ?? null,
    notes: flags.notes ?? null,
    timestamp,
    source: {
      mode: 'import',
      inputPath: path.resolve(inputPath),
      command: null,
      commandTimeoutMs: null,
      rawSha256: rawInfo.sha256,
      rawBytes: rawInfo.bytes,
      host: null,
    },
  });

  const outputPath = flags.output
    ? path.resolve(flags.output)
    : defaultOutputPath(product.id, timestamp);
  await writeRecord(record, outputPath);
  console.log(outputPath);
}

async function doRun(flags, passthrough, timestamp = null) {
  const targetId = flags.target;
  if (!targetId) {
    throw new Error('run requires --target <id>');
  }
  const commandTimeoutMs = parsePositiveInteger(flags['timeout-ms'], DEFAULT_COMMAND_TIMEOUT_MS, '--timeout-ms');
  const resourceTelemetry = {
    enabled: parseResourceTelemetryMode(flags['resource-telemetry'], false, '--resource-telemetry'),
    intervalMs: parsePositiveInteger(
      flags['resource-telemetry-interval-ms'],
      DEFAULT_RESOURCE_TELEMETRY_INTERVAL_MS,
      '--resource-telemetry-interval-ms'
    ),
    includeSamples: parseBooleanFlag(flags['resource-telemetry-samples'], false, '--resource-telemetry-samples'),
    label: targetId,
  };

  const { registry, workloads } = await loadRegistryBundle();
  const { product, harness } = await loadTargetBundle(targetId, registry);
  const commandParts = passthrough.length > 0
    ? passthrough
    : Array.isArray(harness.execution.defaultCommand)
      ? harness.execution.defaultCommand
      : [];

  if (!Array.isArray(commandParts) || commandParts.length === 0) {
    throw new Error(`No run command provided for target "${targetId}".`);
  }

  const runResult = await runCommandCaptureJson(commandParts, {
    timeoutMs: commandTimeoutMs,
    resourceTelemetry,
  });
  const rawResult = runResult.payload;
  const rawInfo = await hashJsonBytes(rawResult);
  const workloadsById = new Map(workloads.workloads.map((item) => [item.id, item]));

  const record = await normalizeRecord({
    product,
    harness,
    rawResult,
    workloadsById,
    workloadId: flags.workload ?? null,
    modelId: flags.model ?? null,
    notes: flags.notes ?? null,
    timestamp,
    resourceTelemetry: runResult.resourceTelemetry,
    source: {
      mode: 'run',
      inputPath: null,
      command: commandParts,
      commandTimeoutMs,
      rawSha256: rawInfo.sha256,
      rawBytes: rawInfo.bytes,
      host: {
        platform: process.platform,
        arch: process.arch,
        nodeVersion: process.version,
        osRelease: typeof os.release === 'function' ? os.release() : null,
        cpuModel: (() => {
          const cpuInfo = typeof os.cpus === 'function' ? os.cpus() : null;
          if (!Array.isArray(cpuInfo) || cpuInfo.length === 0) return null;
          return asNonEmptyStringValue(cpuInfo[0]?.model);
        })(),
      },
    },
  });

  const outputPath = flags.output
    ? path.resolve(flags.output)
    : defaultOutputPath(product.id, timestamp);
  await writeRecord(record, outputPath);
  console.log(outputPath);
}

async function doFixturesRestamp(flags) {
  const checkOnly = parseBooleanFlag(flags.check, false, '--check');
  const fixtureFlag = typeof flags.fixture === 'string' && flags.fixture.trim() !== ''
    ? flags.fixture.trim()
    : null;

  const fixtureAbsPaths = [];
  if (fixtureFlag) {
    const abs = path.isAbsolute(fixtureFlag) ? fixtureFlag : path.resolve(ROOT_DIR, fixtureFlag);
    if (!(await fileExists(abs))) {
      throw new Error(`fixtures-restamp: fixture not found: ${fixtureFlag}`);
    }
    fixtureAbsPaths.push(abs);
  } else {
    const entries = await fs.readdir(FIXTURES_DIR, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isFile()) continue;
      if (!isCommittedCompareFixtureFileName(entry.name)) continue;
      fixtureAbsPaths.push(path.join(FIXTURES_DIR, entry.name));
    }
    fixtureAbsPaths.sort();
  }

  const compareConfig = JSON.parse(await fs.readFile(COMPARE_CONFIG_PATH, 'utf8'));
  const profilesByModelId = new Map(
    (compareConfig.modelProfiles || []).map((profile) => [profile.dopplerModelId, profile])
  );

  const hashCache = new Map();
  const hashSourceOnce = async (source) => {
    if (typeof source !== 'string' || source.trim() === '') return null;
    if (hashCache.has(source)) return hashCache.get(source);
    const value = await hashCompareArtifactSource(source);
    hashCache.set(source, value.sha256);
    return value.sha256;
  };

  let updated = 0;
  let unchanged = 0;
  let failed = 0;

  for (const abs of fixtureAbsPaths) {
    const relPath = toPosixRelative(abs);
    const raw = await fs.readFile(abs, 'utf8');
    const fixture = JSON.parse(raw);

    const dopplerModelId = asNonEmptyStringValue(fixture?.dopplerModelId);
    if (!dopplerModelId) {
      console.error(`[restamp] ${relPath}: fixture is missing dopplerModelId`);
      failed += 1;
      continue;
    }
    const profile = profilesByModelId.get(dopplerModelId);
    if (!profile) {
      console.error(
        `[restamp] ${relPath}: no current compareConfig profile for dopplerModelId="${dopplerModelId}" — profile may have been removed; fixture must be recaptured or deleted.`
      );
      failed += 1;
      continue;
    }

    const fixtureTjs = asNonEmptyStringValue(fixture?.tjsModelId);
    const profileTjs = asNonEmptyStringValue(profile?.defaultTjsModelId);
    if (fixtureTjs && profileTjs && fixtureTjs !== profileTjs) {
      console.error(
        `[restamp] ${relPath}: tjsModelId drift — fixture captured "${fixtureTjs}", current profile default is "${profileTjs}". Recapture required.`
      );
      failed += 1;
      continue;
    }

    const checks = [
      { label: 'benchmarkPolicy', block: fixture?.benchmarkPolicy },
      { label: 'compareConfig', block: fixture?.compareConfig },
      { label: 'metricContract', block: fixture?.metricContract },
    ];
    if (fixture?.harnesses && typeof fixture.harnesses === 'object') {
      for (const harnessKey of Object.keys(fixture.harnesses)) {
        checks.push({
          label: `harnesses.${harnessKey}`,
          block: fixture.harnesses[harnessKey],
          harnessKey,
        });
      }
    }

    const drifts = [];
    let blocked = false;
    for (const check of checks) {
      const source = asNonEmptyStringValue(check.block?.source);
      const declared = asNonEmptyStringValue(check.block?.sourceSha256);
      if (!source || !declared) continue;
      let current;
      try {
        current = await hashSourceOnce(source);
      } catch (err) {
        console.error(`[restamp] ${relPath}: ${check.label} source "${source}" cannot be hashed: ${err.message}`);
        blocked = true;
        break;
      }
      if (current !== declared) {
        drifts.push({ label: check.label, block: check.block, from: declared, to: current });
      }
    }
    if (blocked) {
      failed += 1;
      continue;
    }

    if (drifts.length === 0) {
      console.log(`[restamp] ${relPath}: already up to date`);
      unchanged += 1;
      continue;
    }

    console.log(`[restamp] ${relPath}: ${drifts.length} drift${drifts.length === 1 ? '' : 's'}`);
    for (const drift of drifts) {
      console.log(`  ${drift.label}.sourceSha256: ${drift.from} -> ${drift.to}`);
    }

    if (checkOnly) {
      continue;
    }

    for (const drift of drifts) {
      drift.block.sourceSha256 = drift.to;
    }
    const serialized = JSON.stringify(fixture, null, 2) + '\n';
    await fs.writeFile(abs, serialized, 'utf8');
    updated += 1;
  }

  console.log(`\n[restamp] summary: updated=${updated}, unchanged=${unchanged}, failed=${failed}`);
  if (failed > 0) {
    process.exitCode = 1;
  }
}

async function main() {
  const parsed = parseArgs(process.argv.slice(2));
  const command = parsed.command;
  const rawHelp = parsed.flags.help ?? parsed.flags.h ?? false;
  const helpRequested = parseBooleanFlag(rawHelp, false, '--help');
  if (!command || helpRequested) {
    console.log(usage());
    return;
  }
  const timestamp = parseTimestampValue(parsed.flags.timestamp, '--timestamp');

  if (command === 'list') {
    const { registry } = await loadRegistryBundle();
    printList(registry);
    return;
  }
  if (command === 'validate') {
    await doValidate();
    return;
  }
  if (command === 'capabilities') {
    await doCapabilities(parsed.flags);
    return;
  }
  if (command === 'gap') {
    await doGap(parsed.flags);
    return;
  }
  if (command === 'matrix') {
    await doMatrix(parsed.flags, timestamp);
    return;
  }
  if (command === 'show') {
    await doShow(parsed.flags);
    return;
  }
  if (command === 'import') {
    await doImport(parsed.flags, timestamp);
    return;
  }
  if (command === 'run') {
    await doRun(parsed.flags, parsed.passthrough, timestamp);
    return;
  }
  if (command === 'fixtures-restamp') {
    await doFixturesRestamp(parsed.flags);
    return;
  }

  throw new Error(`Unknown command "${command}"`);
}

main().catch((error) => {
  console.error(`[vendor-bench] ${error.message}`);
  process.exitCode = 1;
});
