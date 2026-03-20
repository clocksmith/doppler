#!/usr/bin/env node

import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, relative, resolve } from 'node:path';

import { loadTrainingWorkloadPack } from '../src/training/workloads.js';
import { sha256Hex } from '../src/utils/sha256.js';

function normalizeString(value) {
  if (value === undefined || value === null) return null;
  const trimmed = String(value).trim();
  return trimmed || null;
}

function parseArgs(argv) {
  const parsed = {
    root: 'src/training/workload-packs',
    registry: 'src/training/workload-packs/registry.json',
    out: null,
    writeRegistry: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const token = String(argv[index] || '');
    if (token === '--root') {
      parsed.root = String(argv[index + 1] || '').trim();
      index += 1;
      continue;
    }
    if (token === '--registry') {
      parsed.registry = String(argv[index + 1] || '').trim();
      index += 1;
      continue;
    }
    if (token === '--out') {
      parsed.out = String(argv[index + 1] || '').trim();
      index += 1;
      continue;
    }
    if (token === '--write-registry') {
      parsed.writeRegistry = true;
      continue;
    }
    throw new Error(`Unknown argument: ${token}`);
  }
  return {
    rootPath: resolve(parsed.root),
    registryPath: resolve(parsed.registry),
    outPath: normalizeString(parsed.out) ? resolve(parsed.out) : null,
    writeRegistry: parsed.writeRegistry,
  };
}

function buildBaselineReportId(id, sha256) {
  return `trn_${id}_${sha256.slice(0, 12)}`;
}

async function collectWorkloadEntries(rootPath, registryPath) {
  const entries = await readdir(rootPath, { withFileTypes: true });
  const files = entries
    .filter((entry) => entry.isFile() && entry.name.endsWith('.json'))
    .map((entry) => join(rootPath, entry.name))
    .filter((filePath) => resolve(filePath) !== registryPath)
    .sort((left, right) => left.localeCompare(right));

  const seenIds = new Set();
  const workloads = [];
  for (const filePath of files) {
    const loaded = await loadTrainingWorkloadPack(filePath);
    if (seenIds.has(loaded.workload.id)) {
      throw new Error(`Duplicate workload id: ${loaded.workload.id}`);
    }
    seenIds.add(loaded.workload.id);
    workloads.push({
      id: loaded.workload.id,
      path: relative(process.cwd(), loaded.absolutePath).replace(/\\/g, '/'),
      sha256: loaded.workloadSha256,
      workloadKind: loaded.workload.kind,
      trainingSchemaVersion: loaded.workload.trainingSchemaVersion,
      seed: loaded.workload.seed,
      checkpointEvery: loaded.workload.checkpointEvery,
      selectionMetric: loaded.workload.selectionMetric,
      selectionGoal: loaded.workload.selectionGoal,
      surfaceSupport: loaded.workload.surfaceSupport,
      baselineReportId: buildBaselineReportId(loaded.workload.id, loaded.workloadSha256),
      claimBoundary: loaded.workload.claimBoundary,
      configHash: loaded.workload.configHash,
    });
  }
  if (workloads.length === 0) {
    throw new Error(`No workload packs found in ${rootPath}.`);
  }
  return workloads;
}

async function readJson(pathValue, label) {
  const raw = await readFile(pathValue, 'utf8');
  try {
    return JSON.parse(raw);
  } catch (error) {
    throw new Error(`${label} is not valid JSON: ${error.message}`);
  }
}

function validateRegistryPayload(registry, workloads, registryPath) {
  if (!registry || typeof registry !== 'object' || Array.isArray(registry)) {
    throw new Error(`${registryPath} must be a JSON object.`);
  }
  if (registry.schemaVersion !== 1) {
    throw new Error(`${registryPath}.schemaVersion must be 1.`);
  }
  if (!Array.isArray(registry.workloads)) {
    throw new Error(`${registryPath}.workloads must be an array.`);
  }
  if (registry.workloads.length !== workloads.length) {
    const registryIds = new Set(registry.workloads.map((entry) => entry.id));
    const scannedIds = new Set(workloads.map((workload) => workload.id));
    const stale = [...registryIds].filter((id) => !scannedIds.has(id));
    const missing = [...scannedIds].filter((id) => !registryIds.has(id));
    const parts = [];
    if (stale.length > 0) parts.push(`stale registry entries: ${stale.join(', ')}`);
    if (missing.length > 0) parts.push(`missing from registry: ${missing.join(', ')}`);
    throw new Error(`registry out of sync (${parts.join('; ')}). Run --write-registry to update.`);
  }
  const byId = new Map(registry.workloads.map((entry) => [entry.id, entry]));
  for (const workload of workloads) {
    const entry = byId.get(workload.id);
    if (!entry) {
      throw new Error(`registry missing workload "${workload.id}".`);
    }
    if (entry.path !== workload.path) {
      throw new Error(`registry path mismatch for "${workload.id}".`);
    }
    if (entry.sha256 !== workload.sha256) {
      throw new Error(`registry sha256 mismatch for "${workload.id}".`);
    }
    if (entry.baselineReportId !== workload.baselineReportId) {
      throw new Error(`registry baselineReportId mismatch for "${workload.id}".`);
    }
    const contractFields = ['workloadKind', 'trainingSchemaVersion', 'seed'];
    for (const field of contractFields) {
      if (entry[field] !== workload[field]) {
        throw new Error(`registry ${field} mismatch for "${workload.id}".`);
      }
    }
  }
}

function buildRegistryPayload(workloads) {
  return {
    schemaVersion: 1,
    artifactType: 'training_workload_registry',
    generatedAt: new Date().toISOString(),
    registryHash: sha256Hex(JSON.stringify(workloads)),
    workloads,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const workloads = await collectWorkloadEntries(args.rootPath, args.registryPath);
  const registryPayload = buildRegistryPayload(workloads);

  if (args.writeRegistry) {
    await mkdir(dirname(args.registryPath), { recursive: true });
    await writeFile(args.registryPath, `${JSON.stringify(registryPayload, null, 2)}\n`, 'utf8');
  } else {
    const existing = await readJson(args.registryPath, relative(process.cwd(), args.registryPath));
    validateRegistryPayload(existing, workloads, args.registryPath);
  }

  const output = {
    ok: true,
    rootPath: relative(process.cwd(), args.rootPath).replace(/\\/g, '/'),
    registryPath: relative(process.cwd(), args.registryPath).replace(/\\/g, '/'),
    workloadCount: workloads.length,
    workloads,
  };
  if (args.outPath) {
    await mkdir(dirname(args.outPath), { recursive: true });
    await writeFile(args.outPath, `${JSON.stringify(output, null, 2)}\n`, 'utf8');
  } else {
    console.log(JSON.stringify(output, null, 2));
  }
}

await main();
