#!/usr/bin/env node

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, relative, resolve } from 'node:path';

function normalizeString(value) {
  if (value === undefined || value === null) return null;
  const trimmed = String(value).trim();
  return trimmed || null;
}

function parseArgs(argv) {
  const parsed = {
    registry: 'src/experimental/training/workload-packs/registry.json',
    out: 'reports/training/report-ids/latest.json',
  };
  for (let i = 0; i < argv.length; i += 1) {
    const token = String(argv[i] || '');
    if (token === '--registry') {
      parsed.registry = String(argv[i + 1] || '').trim();
      if (!parsed.registry) {
        throw new Error('Missing value for --registry');
      }
      i += 1;
      continue;
    }
    if (token === '--out') {
      parsed.out = String(argv[i + 1] || '').trim();
      if (!parsed.out) {
        throw new Error('Missing value for --out');
      }
      i += 1;
      continue;
    }
    throw new Error(`Unknown argument: ${token}`);
  }
  return {
    registryPath: resolve(parsed.registry),
    outPath: resolve(parsed.out),
  };
}

async function readJson(pathValue, label) {
  const raw = await readFile(pathValue, 'utf8');
  try {
    return JSON.parse(raw);
  } catch (error) {
    throw new Error(`${label} is not valid JSON: ${error.message}`);
  }
}

function assertRegistryPayload(registry, registryPath) {
  if (!registry || typeof registry !== 'object' || Array.isArray(registry)) {
    throw new Error(`${registryPath} must be a JSON object.`);
  }
  if (registry.schemaVersion !== 1) {
    throw new Error(`${registryPath}.schemaVersion must be 1.`);
  }
  if (!Array.isArray(registry.workloads)) {
    throw new Error(`${registryPath}.workloads must be an array.`);
  }
  return registry.workloads.map((entry, index) => {
    const id = normalizeString(entry?.id);
    const path = normalizeString(entry?.path);
    const sha256 = normalizeString(entry?.sha256);
    const baselineReportId = normalizeString(entry?.baselineReportId);
    const claimBoundary = normalizeString(entry?.claimBoundary);
    if (!id || !path || !sha256 || !baselineReportId || !claimBoundary) {
      throw new Error(
        `${registryPath}.workloads[${index}] must include id, path, sha256, baselineReportId, claimBoundary.`
      );
    }
    return {
      id,
      path,
      sha256,
      baselineReportId,
      claimBoundary,
    };
  });
}

function buildPublication(registryPath, workloads) {
  return {
    schemaVersion: 1,
    artifactType: 'training_report_id_index',
    generatedAt: new Date().toISOString(),
    sourceRegistry: relative(process.cwd(), registryPath).replace(/\\/g, '/'),
    claimCount: workloads.length,
    claims: workloads.map((entry) => ({
      reportId: entry.baselineReportId,
      workloadId: entry.id,
      workloadPath: entry.path,
      workloadSha256: entry.sha256,
      claimBoundary: entry.claimBoundary,
      status: 'published',
    })),
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const registry = await readJson(args.registryPath, relative(process.cwd(), args.registryPath));
  const workloads = assertRegistryPayload(registry, args.registryPath);
  const publication = buildPublication(args.registryPath, workloads);
  await mkdir(dirname(args.outPath), { recursive: true });
  await writeFile(args.outPath, `${JSON.stringify(publication, null, 2)}\n`, 'utf8');
  console.log(`[training-report-ids] wrote ${args.outPath}`);
}

await main();
