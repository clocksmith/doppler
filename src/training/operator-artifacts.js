import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, relative, resolve } from 'node:path';

import { sha256Hex } from '../utils/sha256.js';
import { serializeTrainingWorkloadLock } from './workloads.js';

function stableSortObject(value) {
  if (Array.isArray(value)) {
    return value.map((entry) => stableSortObject(entry));
  }
  if (!value || typeof value !== 'object') {
    return value;
  }
  const sorted = {};
  for (const key of Object.keys(value).sort()) {
    sorted[key] = stableSortObject(value[key]);
  }
  return sorted;
}

function stableJson(value) {
  return JSON.stringify(stableSortObject(value));
}

export function normalizeTrainingTimestamp(value = null) {
  const date = value instanceof Date
    ? value
    : (typeof value === 'string' && value.trim() ? new Date(value) : new Date());
  return date.toISOString().replace(/[:]/g, '-');
}

export async function createTrainingRunLayout({ kind, workloadId, timestamp = null }) {
  const normalizedKind = String(kind || '').trim();
  const normalizedWorkloadId = String(workloadId || '').trim();
  if (!normalizedKind || !normalizedWorkloadId) {
    throw new Error('createTrainingRunLayout requires kind and workloadId.');
  }
  const ts = normalizeTrainingTimestamp(timestamp);
  const runRoot = resolve('reports', 'training', normalizedKind, normalizedWorkloadId, ts);
  const directories = {
    runRoot,
    logs: join(runRoot, 'logs'),
    checkpoints: join(runRoot, 'checkpoints'),
    eval: join(runRoot, 'eval'),
    scoreboard: join(runRoot, 'scoreboard'),
    exports: join(runRoot, 'exports'),
    compare: join(runRoot, 'compare'),
    qualityGate: join(runRoot, 'quality-gate'),
  };
  await Promise.all(Object.values(directories).map((dirPath) => mkdir(dirPath, { recursive: true })));
  return directories;
}

export async function writeJsonArtifact(filePath, payload) {
  const absolutePath = resolve(String(filePath));
  const json = `${JSON.stringify(payload, null, 2)}\n`;
  await mkdir(dirname(absolutePath), { recursive: true });
  await writeFile(absolutePath, json, 'utf8');
  return {
    path: absolutePath,
    sha256: sha256Hex(json),
    relativePath: relative(process.cwd(), absolutePath),
  };
}

export async function writeNdjsonRow(filePath, row) {
  const absolutePath = resolve(String(filePath));
  let existing = '';
  try {
    existing = await readFile(absolutePath, 'utf8');
  } catch (error) {
    if (error?.code !== 'ENOENT') {
      throw error;
    }
  }
  const serialized = `${existing}${JSON.stringify(row)}\n`;
  await mkdir(dirname(absolutePath), { recursive: true });
  await writeFile(absolutePath, serialized, 'utf8');
  return {
    path: absolutePath,
    sha256: sha256Hex(serialized),
    relativePath: relative(process.cwd(), absolutePath),
  };
}

export async function writeWorkloadLock(layout, loadedWorkload) {
  const filePath = join(layout.runRoot, 'workload.lock.json');
  const payload = JSON.parse(serializeTrainingWorkloadLock(loadedWorkload));
  return writeJsonArtifact(filePath, payload);
}

export async function writeRunContract(layout, payload) {
  return writeJsonArtifact(join(layout.runRoot, 'run_contract.json'), payload);
}

export function buildArtifactBase({
  artifactType,
  reportId,
  workload,
  workloadPath,
  workloadSha256,
  datasetPath,
  datasetHash,
  baseModelId,
  teacherModelId = null,
  studentModelId = null,
  stage = null,
  checkpointStep = null,
  parentArtifacts = [],
  runtime = 'node',
  surface = 'node',
  claimBoundary,
  configHash,
}) {
  return {
    artifactType,
    schemaVersion: 1,
    reportId,
    workloadId: workload.id,
    workloadPath,
    workloadSha256,
    configHash,
    datasetPath: datasetPath || null,
    datasetHash: datasetHash || null,
    baseModelId: baseModelId || null,
    teacherModelId: teacherModelId || null,
    studentModelId: studentModelId || null,
    stage,
    checkpointStep,
    parentArtifacts,
    generatedAt: new Date().toISOString(),
    runtime,
    surface,
    claimBoundary,
  };
}

export function hashArtifactPayload(payload) {
  return sha256Hex(stableJson(payload));
}
