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
const REGISTRY_DIR = path.join(ROOT_DIR, 'benchmarks', 'competitors');
const REGISTRY_PATH = path.join(REGISTRY_DIR, 'registry.json');
const WORKLOADS_PATH = path.join(REGISTRY_DIR, 'workloads.json');
const CAPABILITIES_PATH = path.join(REGISTRY_DIR, 'capabilities.json');
const RESULTS_DIR = path.join(REGISTRY_DIR, 'results');

function usage() {
  return [
    'Usage:',
    '  node tools/competitor-bench.js list',
    '  node tools/competitor-bench.js validate',
    '  node tools/competitor-bench.js capabilities [--target <id>]',
    '  node tools/competitor-bench.js gap --base <id> --target <id>',
    '  node tools/competitor-bench.js show --target <id>',
    '  node tools/competitor-bench.js import --target <id> --input <raw.json> [--output <result.json>] [--workload <id>] [--model <id>] [--notes <text>]',
    '  node tools/competitor-bench.js run --target <id> [--output <result.json>] [--workload <id>] [--model <id>] [--notes <text>] -- <command ...>',
    '',
    'Notes:',
    '  - `run` expects command stdout to be a single JSON object.',
    '  - `import` and `run` write normalized records to benchmarks/competitors/results/ by default.',
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
  assertRegistryShape(registry);
  assertWorkloadsShape(workloads);
  collectUniqueIds(registry.products.map((item) => item.id), 'product');
  collectUniqueIds(workloads.workloads.map((item) => item.id), 'workload');
  return { registry, workloads };
}

async function loadCapabilitiesBundle(registry) {
  const capabilities = await readJson(CAPABILITIES_PATH);
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

  return {
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

async function runCommandCaptureJson(commandParts) {
  if (!Array.isArray(commandParts) || commandParts.length === 0) {
    throw new Error('No command provided for run mode');
  }
  const [command, ...args] = commandParts;
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: ROOT_DIR,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: process.env,
    });

    const stdoutChunks = [];
    const stderrChunks = [];
    child.stdout.on('data', (chunk) => stdoutChunks.push(chunk));
    child.stderr.on('data', (chunk) => stderrChunks.push(chunk));
    child.on('error', (error) => reject(error));
    child.on('close', (code) => {
      const stdout = Buffer.concat(stdoutChunks).toString('utf8').trim();
      const stderr = Buffer.concat(stderrChunks).toString('utf8').trim();
      if (code !== 0) {
        reject(new Error(`Command failed (${code}): ${stderr || commandParts.join(' ')}`));
        return;
      }
      try {
        resolve(JSON.parse(stdout));
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

  const rawResult = await runCommandCaptureJson(commandParts);
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
  console.error(`[competitor-bench] ${error.message}`);
  process.exitCode = 1;
});
