#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, '..');
const REGISTRY_DIR = path.join(ROOT_DIR, 'benchmarks', 'vendors');
const REGISTRY_PATH = path.join(REGISTRY_DIR, 'registry.json');
const WORKLOADS_PATH = path.join(REGISTRY_DIR, 'workloads.json');
const CAPABILITIES_PATH = path.join(REGISTRY_DIR, 'capabilities.json');
const RESULTS_DIR = path.join(REGISTRY_DIR, 'results');
const SCHEMA_DIR = path.join(REGISTRY_DIR, 'schema');
const REGISTRY_SCHEMA_PATH = path.join(SCHEMA_DIR, 'registry.schema.json');
const CAPABILITIES_SCHEMA_PATH = path.join(SCHEMA_DIR, 'capabilities.schema.json');
const HARNESS_SCHEMA_PATH = path.join(SCHEMA_DIR, 'harness.schema.json');
const RESULT_SCHEMA_PATH = path.join(SCHEMA_DIR, 'result.schema.json');
const DEFAULT_COMMAND_TIMEOUT_MS = 600_000;

function usage() {
  return [
    'Usage:',
    '  node tools/vendor-bench.js list',
    '  node tools/vendor-bench.js validate',
    '  node tools/vendor-bench.js capabilities [--target <id>]',
    '  node tools/vendor-bench.js gap --base <id> --target <id>',
    '  node tools/vendor-bench.js show --target <id>',
    '  node tools/vendor-bench.js import --target <id> --input <raw.json> [--output <result.json>] [--workload <id>] [--model <id>] [--notes <text>]',
    '  node tools/vendor-bench.js run --target <id> [--timeout-ms <ms>] [--output <result.json>] [--workload <id>] [--model <id>] [--notes <text>] -- <command ...>',
    '  --timeout-ms <ms>           Command timeout in milliseconds (default: 600000)',
    '',
    'Notes:',
    '  - `run` expects command stdout to include a JSON object payload.',
    '  - `import` and `run` write normalized records to benchmarks/vendors/results/ by default.',
  ].join('\n');
}

function parseArgs(argv) {
  const out = {
    command: argv[0] ?? null,
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

    const key = token.slice(2);
    if (key === 'help' || key === 'h') {
      out.flags.help = 'true';
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

function parsePositiveInt(value, fallback, label) {
  if (value == null || value === '') return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || !Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive integer`);
  }
  return parsed;
}

function parseJsonFromStdout(stdout, label) {
  const normalized = String(stdout == null ? '' : stdout);
  if (!normalized.trim()) {
    throw new Error(`Command produced no output for ${label}`);
  }

  const tail = (text, maxChars = 2000) => {
    const str = String(text || '');
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
      if (typeof value !== 'boolean') {
        throw new Error(`capabilities target "${entry.id}" bench.features.${key} must be boolean`);
      }
    }
    for (const key of profileFeatureIds) {
      const value = profileFeatures[key];
      if (typeof value !== 'boolean') {
        throw new Error(`capabilities target "${entry.id}" profile.features.${key} must be boolean`);
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
  if (record.notes !== null && typeof record.notes !== 'string') {
    throw new Error('record.notes must be a string or null');
  }
}

function toIsoTimestamp() {
  return new Date().toISOString();
}

function toFileTimestamp() {
  return toIsoTimestamp()
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

  const selectedWorkload = workloadId ? workloadsById.get(workloadId) || null : null;
  if (workloadId && !selectedWorkload) {
    throw new Error(`Unknown workload id: ${workloadId}`);
  }

  const resolvedModelId = modelId || (metadata.model == null ? null : String(metadata.model));
  const harnessVersion = metadata.version == null ? null : String(metadata.version);

  const record = {
    schemaVersion: 1,
    timestamp: toIsoTimestamp(),
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
    notes: notes ?? null,
    metadata,
  };
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

function defaultOutputPath(targetId) {
  return path.join(RESULTS_DIR, `${targetId}-${toFileTimestamp()}.json`);
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
    if (value === true) count += 1;
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
    if (baseFeatures?.[key] === true && targetFeatures?.[key] !== true) {
      missing.push(key);
    }
  }
  return missing.sort();
}

function listExtraFeatures(baseFeatures, targetFeatures) {
  const extra = [];
  const keys = new Set([...Object.keys(baseFeatures || {}), ...Object.keys(targetFeatures || {})]);
  for (const key of keys) {
    if (baseFeatures?.[key] !== true && targetFeatures?.[key] === true) {
      extra.push(key);
    }
  }
  return extra.sort();
}

async function runCommandCaptureJson(commandParts, options = {}) {
  if (!Array.isArray(commandParts) || commandParts.length === 0) {
    throw new Error('No command provided for run mode');
  }
  const [command, ...args] = commandParts;
  const timeoutMs = parsePositiveInt(options.timeoutMs, DEFAULT_COMMAND_TIMEOUT_MS, '--timeout-ms');
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

async function doImport(flags) {
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
    source: {
      mode: 'import',
      inputPath: path.resolve(inputPath),
      command: null,
      commandTimeoutMs: null,
      rawSha256: rawInfo.sha256,
      rawBytes: rawInfo.bytes,
    },
  });

  const outputPath = flags.output
    ? path.resolve(flags.output)
    : defaultOutputPath(product.id);
  await writeRecord(record, outputPath);
  console.log(outputPath);
}

async function doRun(flags, passthrough) {
  const targetId = flags.target;
  if (!targetId) {
    throw new Error('run requires --target <id>');
  }
  const commandTimeoutMs = parsePositiveInt(flags['timeout-ms'], DEFAULT_COMMAND_TIMEOUT_MS, '--timeout-ms');

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
    source: {
      mode: 'run',
      inputPath: null,
      command: commandParts,
      commandTimeoutMs,
      rawSha256: rawInfo.sha256,
      rawBytes: rawInfo.bytes,
    },
  });

  const outputPath = flags.output
    ? path.resolve(flags.output)
    : defaultOutputPath(product.id);
  await writeRecord(record, outputPath);
  console.log(outputPath);
}

async function main() {
  const parsed = parseArgs(process.argv.slice(2));
  const command = parsed.command;
  const helpRequested = parsed.flags.help === 'true';
  if (!command || helpRequested) {
    console.log(usage());
    return;
  }

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
  if (command === 'show') {
    await doShow(parsed.flags);
    return;
  }
  if (command === 'import') {
    await doImport(parsed.flags);
    return;
  }
  if (command === 'run') {
    await doRun(parsed.flags, parsed.passthrough);
    return;
  }

  throw new Error(`Unknown command "${command}"`);
}

main().catch((error) => {
  console.error(`[vendor-bench] ${error.message}`);
  process.exitCode = 1;
});
