#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, relative, resolve } from 'node:path';

function normalizeString(value) {
  if (value === undefined || value === null) return null;
  const trimmed = String(value).trim();
  return trimmed || null;
}

function parseInteger(value, label, minimum = null) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed)) {
    throw new Error(`${label} must be an integer.`);
  }
  if (minimum != null && parsed < minimum) {
    throw new Error(`${label} must be >= ${minimum}.`);
  }
  return parsed;
}

function parseArgs(argv) {
  const parsed = {
    root: 'tools/configs/training-workloads',
    registry: 'tools/configs/training-workloads/registry.json',
    out: null,
    writeRegistry: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = String(argv[i] || '');
    if (token === '--root') {
      parsed.root = String(argv[i + 1] || '').trim();
      i += 1;
      continue;
    }
    if (token === '--registry') {
      parsed.registry = String(argv[i + 1] || '').trim();
      i += 1;
      continue;
    }
    if (token === '--out') {
      parsed.out = String(argv[i + 1] || '').trim();
      i += 1;
      continue;
    }
    if (token === '--write-registry') {
      parsed.writeRegistry = true;
      continue;
    }
    throw new Error(`Unknown argument: ${token}`);
  }
  const rootPath = resolve(parsed.root || 'tools/configs/training-workloads');
  const registryPath = resolve(parsed.registry || join(rootPath, 'registry.json'));
  return {
    rootPath,
    registryPath,
    outPath: normalizeString(parsed.out) ? resolve(parsed.out) : null,
    writeRegistry: parsed.writeRegistry,
  };
}

function hashSha256(input) {
  return createHash('sha256').update(input).digest('hex');
}

async function readJson(pathValue, label) {
  const raw = await readFile(pathValue, 'utf8');
  try {
    return { raw, value: JSON.parse(raw) };
  } catch (error) {
    throw new Error(`${label} is not valid JSON: ${error.message}`);
  }
}

function ensureArray(value, label) {
  if (!Array.isArray(value)) {
    throw new Error(`${label} must be an array.`);
  }
  return value;
}

function ensureBenchRun(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return {
    warmupRuns: parseInteger(value.warmupRuns, `${label}.warmupRuns`, 0),
    timedRuns: parseInteger(value.timedRuns, `${label}.timedRuns`, 1),
  };
}

function buildBaselineReportId(id, sha256) {
  return `trn_${id}_${sha256.slice(0, 12)}`;
}

function classifyWorkload(payload) {
  const id = normalizeString(payload.id) || '';
  if (id.startsWith('distill-')) return 'distill';
  return 'ul';
}

function normalizeWorkloadPayload(payload, contextLabel) {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error(`${contextLabel} must be a JSON object.`);
  }
  const schemaVersion = parseInteger(payload.schemaVersion, `${contextLabel}.schemaVersion`, 1);
  if (schemaVersion !== 1) {
    throw new Error(`${contextLabel}.schemaVersion must be 1.`);
  }
  const id = normalizeString(payload.id);
  if (!id) {
    throw new Error(`${contextLabel}.id is required.`);
  }
  const description = normalizeString(payload.description);
  if (!description) {
    throw new Error(`${contextLabel}.description is required.`);
  }
  const seed = parseInteger(payload.seed, `${contextLabel}.seed`);
  const trainingSchemaVersion = parseInteger(payload.trainingSchemaVersion, `${contextLabel}.trainingSchemaVersion`, 1);
  if (trainingSchemaVersion !== 1) {
    throw new Error(`${contextLabel}.trainingSchemaVersion must be 1.`);
  }
  const trainingBenchSteps = parseInteger(payload.trainingBenchSteps, `${contextLabel}.trainingBenchSteps`, 1);
  const trainingTests = ensureArray(payload.trainingTests, `${contextLabel}.trainingTests`)
    .map((entry, index) => {
      const testId = normalizeString(entry);
      if (!testId) {
        throw new Error(`${contextLabel}.trainingTests[${index}] must be a non-empty string.`);
      }
      return testId;
    });
  if (trainingTests.length === 0) {
    throw new Error(`${contextLabel}.trainingTests must contain at least one suite id.`);
  }
  const benchRun = ensureBenchRun(payload.benchRun, `${contextLabel}.benchRun`);
  const workloadKind = classifyWorkload(payload);
  const normalized = {
    schemaVersion,
    id,
    description,
    seed,
    trainingSchemaVersion,
    benchRun,
    trainingBenchSteps,
    trainingTests,
    workloadKind,
  };
  if (workloadKind === 'distill') {
    normalized.teacherModelId = normalizeString(payload.teacherModelId);
    normalized.studentModelId = normalizeString(payload.studentModelId);
    normalized.distillDatasetId = normalizeString(payload.distillDatasetId);
    normalized.distillLanguagePair = normalizeString(payload.distillLanguagePair);
    if (!normalized.teacherModelId || !normalized.studentModelId || !normalized.distillDatasetId || !normalized.distillLanguagePair) {
      throw new Error(`${contextLabel} distill workloads require teacherModelId, studentModelId, distillDatasetId, and distillLanguagePair.`);
    }
  }
  return normalized;
}

async function collectWorkloads(rootPath, registryPath) {
  const entries = await readdir(rootPath, { withFileTypes: true });
  const files = entries
    .filter((entry) => entry.isFile() && entry.name.endsWith('.json'))
    .map((entry) => join(rootPath, entry.name))
    .filter((filePath) => resolve(filePath) !== registryPath)
    .sort((left, right) => left.localeCompare(right));

  if (files.length === 0) {
    throw new Error(`No workload packs found in ${rootPath}.`);
  }

  const seenIds = new Set();
  const workloads = [];
  for (const filePath of files) {
    const parsed = await readJson(filePath, relative(process.cwd(), filePath));
    const normalized = normalizeWorkloadPayload(
      parsed.value,
      relative(process.cwd(), filePath)
    );
    if (seenIds.has(normalized.id)) {
      throw new Error(`Duplicate workload id: ${normalized.id}`);
    }
    seenIds.add(normalized.id);
    const sha256 = hashSha256(parsed.raw);
    workloads.push({
      id: normalized.id,
      path: relative(process.cwd(), filePath).replace(/\\/g, '/'),
      sha256,
      workloadKind: normalized.workloadKind,
      trainingSchemaVersion: normalized.trainingSchemaVersion,
      seed: normalized.seed,
      trainingBenchSteps: normalized.trainingBenchSteps,
      trainingTests: normalized.trainingTests,
      benchRun: normalized.benchRun,
      baselineReportId: buildBaselineReportId(normalized.id, sha256),
      claimBoundary: normalized.workloadKind === 'distill'
        ? 'Practical distill workflow quality traceability.'
        : 'Practical UL workflow quality traceability.',
    });
  }
  return workloads;
}

function validateRegistryPayload(registry, workloads, registryPath) {
  if (!registry || typeof registry !== 'object' || Array.isArray(registry)) {
    throw new Error(`${registryPath} must be a JSON object.`);
  }
  const schemaVersion = parseInteger(registry.schemaVersion, `${registryPath}.schemaVersion`, 1);
  if (schemaVersion !== 1) {
    throw new Error(`${registryPath}.schemaVersion must be 1.`);
  }
  const entries = ensureArray(registry.workloads, `${registryPath}.workloads`);
  const byId = new Map(entries.map((entry) => [normalizeString(entry?.id), entry]));
  for (const workload of workloads) {
    const entry = byId.get(workload.id);
    if (!entry) {
      throw new Error(`registry missing workload id "${workload.id}".`);
    }
    const pathValue = normalizeString(entry.path);
    const shaValue = normalizeString(entry.sha256);
    const reportId = normalizeString(entry.baselineReportId);
    if (pathValue !== workload.path) {
      throw new Error(`registry path mismatch for "${workload.id}": expected ${workload.path}, got ${pathValue || 'null'}.`);
    }
    if (shaValue !== workload.sha256) {
      throw new Error(`registry sha256 mismatch for "${workload.id}": expected ${workload.sha256}, got ${shaValue || 'null'}.`);
    }
    if (reportId !== workload.baselineReportId) {
      throw new Error(`registry baselineReportId mismatch for "${workload.id}": expected ${workload.baselineReportId}, got ${reportId || 'null'}.`);
    }
  }
}

function buildRegistryPayload(workloads) {
  return {
    schemaVersion: 1,
    artifactType: 'training_workload_registry',
    generatedAt: new Date().toISOString(),
    workloads,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const workloads = await collectWorkloads(args.rootPath, args.registryPath);
  const registryPayload = buildRegistryPayload(workloads);

  if (args.writeRegistry) {
    await mkdir(dirname(args.registryPath), { recursive: true });
    await writeFile(args.registryPath, `${JSON.stringify(registryPayload, null, 2)}\n`, 'utf8');
  } else {
    const registryFile = await readJson(args.registryPath, relative(process.cwd(), args.registryPath));
    validateRegistryPayload(registryFile.value, workloads, args.registryPath);
  }

  const output = {
    ok: true,
    rootPath: relative(process.cwd(), args.rootPath).replace(/\\/g, '/'),
    registryPath: relative(process.cwd(), args.registryPath).replace(/\\/g, '/'),
    workloadCount: workloads.length,
    workloads: workloads.map((entry) => ({
      id: entry.id,
      path: entry.path,
      sha256: entry.sha256,
      baselineReportId: entry.baselineReportId,
    })),
  };

  if (args.outPath) {
    await mkdir(dirname(args.outPath), { recursive: true });
    await writeFile(args.outPath, `${JSON.stringify(output, null, 2)}\n`, 'utf8');
  }

  console.log(JSON.stringify(output, null, 2));
}

await main();
