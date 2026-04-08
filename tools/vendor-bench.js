#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import process from 'node:process';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

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
    '  node tools/vendor-bench.js run --target <id> [--timeout-ms <ms>] [--timestamp <iso|ms>] [--output <result.json>] [--workload <id>] [--model <id>] [--notes <text>] -- <command ...>',
    '  --timeout-ms <ms>           Command timeout in milliseconds (default: 600000)',
    '  --timestamp <iso|ms>         Override deterministic timestamp for generated record/matrix timestamps',
    '  --include-local-results      Include benchmarks/vendors/results/*.json in matrix discovery (default: fixtures only)',
    '  --strict-compare-artifacts   Fail matrix generation on any auto-discovered compare artifact parse error',
    '',
    'Notes:',
    '  - `run` expects command stdout to include a JSON object payload.',
    '  - `import` and `run` write normalized records to benchmarks/vendors/results/ by default.',
  ].join('\n');
}

function parseArgs(argv) {
  const booleanFlags = new Set(['help', 'h', 'include-local-results', 'strict-compare-artifacts']);
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
    metricIds: metrics
      .map((entry) => (typeof entry?.id === 'string' ? entry.id.trim() : ''))
      .filter(Boolean),
  };
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
  const candidates = [
    ['compute', 'parity'],
    ['compute', 'throughput'],
    ['warm'],
    ['cold'],
  ];
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
      return {
        id: chain.join('/'),
        payload: cursor,
      };
    }
  }
  return null;
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

function readCompareMetric(payload, metricId) {
  if (!payload || typeof payload !== 'object') return null;
  const metricPaths = [
    ['result', 'timing', metricId],
    ['timing', metricId],
    ['result', metricId],
    [metricId],
  ];
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

async function maybeLoadCompareResultSummary(compareResultPath, compareMetricIds = null) {
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
  return {
    path: repoPath,
    timestamp: typeof report.timestamp === 'string' ? report.timestamp : null,
    mode: typeof report.mode === 'string' ? report.mode : null,
    section: section?.id ?? null,
    decodeProfile: typeof report.decodeProfile === 'string' ? report.decodeProfile : null,
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
    dopplerKernelPath: typeof report.dopplerKernelPath === 'string' ? report.dopplerKernelPath : null,
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
  };
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
  const values = [
    asNonEmptyString(host.platform) || 'unknown',
    asNonEmptyString(host.arch) || 'unknown',
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
      const summary = await maybeLoadCompareResultSummary(candidatePath, compareMetricIds);
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
  const absolutePath = path.isAbsolute(sourcePath)
    ? sourcePath
    : path.resolve(ROOT_DIR, sourcePath);
  if (!(await fileExists(absolutePath))) {
    throw new Error(`compare artifact source is missing: ${sourcePath}`);
  }
  const raw = await fs.readFile(absolutePath, 'utf8');
  const hashInfo = hashTextBytes(raw);
  return {
    absolutePath,
    sha256: hashInfo.sha256,
  };
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
  return `${gpuValue}; ${backendValue}; ${osValue}; ${browserValue}`;
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
  lines.push('| Workload ID | Model | Prefill | Decode | Sampling | Runtime (GPU/Backend/OS/Browser) | Date |');
  lines.push('|---|---|---:|---:|---|---|---|');
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
      const dateCell = selectedRun && typeof selectedRun.timestamp === 'string' && selectedRun.timestamp.length >= 10
        ? markdownTableCell(selectedRun.timestamp.slice(0, 10))
        : (selectedRun ? 'captured' : 'not captured');
      lines.push(
        `| ${workloadIdCell} | ${modelCell} | `
        + `${workload.prefillTokens ?? ''} | ${workload.decodeTokens ?? ''} | `
        + `${markdownTableCell(formatSamplingLabel(workload.sampling))} | ${runtimeComboCell} | ${dateCell} |`
      );
    }
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
  lines.push('');
  return `${lines.join('\n')}\n`;
}

async function doMatrix(flags, timestamp = null) {
  const { registry, workloads } = await loadRegistryBundle();
  const capabilities = await loadCapabilitiesBundle(registry);
  const compareConfig = await loadCompareConfigBundle();
  const compareMetricBundle = await loadCompareMetricBundle();
  const catalog = await loadModelCatalogBundle();

  const compareResultFlag = flags['compare-result'] ?? null;
  const includeLocalResults = parseBooleanFlag(
    flags['include-local-results'],
    false,
    '--include-local-results'
  );
  const strictCompareArtifacts = parseBooleanFlag(
    flags['strict-compare-artifacts'],
    true,
    '--strict-compare-artifacts'
  );
  const {
    compareResults,
    latestCompareResult,
    droppedCompareArtifacts,
  } = await loadCompareResultSummaries({
    compareResultFlag,
    compareMetricIds: compareMetricBundle.metricIds,
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
    return modelId != null && coveredModelIdSet.has(modelId);
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
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: ROOT_DIR,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: process.env,
    });

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
      reject(error);
    });
    child.on('close', (code) => {
      if (timeoutHandle) clearTimeout(timeoutHandle);
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
        resolve(payload);
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
  await loadCapabilitiesBundle(registry);
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

  const rawResult = await runCommandCaptureJson(commandParts, { timeoutMs: commandTimeoutMs });
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

  throw new Error(`Unknown command "${command}"`);
}

main().catch((error) => {
  console.error(`[vendor-bench] ${error.message}`);
  process.exitCode = 1;
});
