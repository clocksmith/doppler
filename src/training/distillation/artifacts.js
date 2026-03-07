import { mkdir } from 'node:fs/promises';
import { join, resolve } from 'node:path';

import {
  buildArtifactBase,
  createTrainingRunLayout,
  hashArtifactPayload,
  writeJsonArtifact,
  writeRunContract,
  writeWorkloadLock,
} from '../operator-artifacts.js';

function toReportId(prefix, workloadId, suffix) {
  return `${prefix}_${workloadId}_${suffix}`;
}

export async function createDistillationRunArtifacts(options) {
  const loadedWorkload = options.loadedWorkload;
  const workload = loadedWorkload?.workload;
  if (!workload || workload.kind !== 'distill') {
    throw new Error('createDistillationRunArtifacts requires a distill workload pack.');
  }
  const layout = options.runRoot
    ? {
      runRoot: resolve(String(options.runRoot)),
      logs: join(resolve(String(options.runRoot)), 'logs'),
      checkpoints: join(resolve(String(options.runRoot)), 'checkpoints'),
      eval: join(resolve(String(options.runRoot)), 'eval'),
      scoreboard: join(resolve(String(options.runRoot)), 'scoreboard'),
      exports: join(resolve(String(options.runRoot)), 'exports'),
      compare: join(resolve(String(options.runRoot)), 'compare'),
      qualityGate: join(resolve(String(options.runRoot)), 'quality-gate'),
    }
    : await createTrainingRunLayout({
      kind: 'distill',
      workloadId: workload.id,
      timestamp: options.timestamp || null,
    });
  await Promise.all(
    Object.values(layout).map((dirPath) => mkdir(dirPath, { recursive: true }))
  );
  const runContractPayload = {
    artifactType: 'training_run_contract',
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    workloadId: workload.id,
    workloadPath: loadedWorkload.absolutePath,
    workloadSha256: loadedWorkload.workloadSha256,
    configHash: workload.configHash,
    claimBoundary: workload.claimBoundary,
    kind: workload.kind,
    stagePlan: workload.pipeline.stagePlan,
    datasetPath: workload.datasetPath,
    evalDatasets: workload.evalDatasets,
    surfaceSupport: workload.surfaceSupport,
  };
  const runContract = await writeRunContract(layout, runContractPayload);
  const workloadLock = await writeWorkloadLock(layout, loadedWorkload);
  return {
    layout,
    runContract,
    workloadLock,
    runContractPayload,
  };
}

export async function writeDistillStageManifest(layout, payload) {
  const stageId = String(payload?.stageId || payload?.stage || 'stage').trim();
  const filePath = join(layout.checkpoints, stageId, 'distill_stage_manifest.json');
  return writeJsonArtifact(filePath, payload);
}

export async function writeDistillCheckpointMetadata(layout, stageId, checkpointId, payload) {
  const filePath = join(layout.checkpoints, stageId, checkpointId, 'checkpoint.json');
  return writeJsonArtifact(filePath, payload);
}

export async function writeDistillCheckpointComplete(layout, stageId, checkpointId, payload) {
  const filePath = join(layout.checkpoints, stageId, checkpointId, 'checkpoint.complete.json');
  return writeJsonArtifact(filePath, payload);
}

export async function writeDistillEvalReport(layout, payload) {
  const stageId = String(payload?.stage || 'stage').trim();
  const checkpointId = String(payload?.checkpointId || 'checkpoint').trim();
  const evalDatasetId = String(payload?.evalDatasetId || 'eval').trim();
  const filePath = join(layout.eval, stageId, `${checkpointId}__${evalDatasetId}.json`);
  return writeJsonArtifact(filePath, payload);
}

export async function writeDistillCompareReport(layout, payload) {
  return writeJsonArtifact(join(layout.compare, 'compare.json'), payload);
}

export async function writeDistillQualityGateReport(layout, payload) {
  return writeJsonArtifact(join(layout.qualityGate, 'quality-gate.json'), payload);
}

export function buildDistillArtifactBase(loadedWorkload, options) {
  const workload = loadedWorkload.workload;
  const checkpointStep = Number.isInteger(options.checkpointStep)
    ? options.checkpointStep
    : null;
  const reportId = toReportId(
    options.prefix || 'dst',
    workload.id,
    `${options.stage || 'stage'}_${checkpointStep == null ? 'final' : String(checkpointStep).padStart(6, '0')}`
  );
  const payload = buildArtifactBase({
    artifactType: options.artifactType,
    reportId,
    workload,
    workloadPath: loadedWorkload.absolutePath,
    workloadSha256: loadedWorkload.workloadSha256,
    datasetPath: options.datasetPath || workload.datasetPath,
    datasetHash: options.datasetHash || null,
    baseModelId: workload.baseModelId,
    teacherModelId: workload.teacherModelId,
    studentModelId: workload.studentModelId,
    stage: options.stage || null,
    checkpointStep,
    parentArtifacts: options.parentArtifacts || [],
    runtime: options.runtime || 'node',
    surface: options.surface || 'node',
    claimBoundary: workload.claimBoundary,
    configHash: options.configHash || workload.configHash,
  });
  return {
    ...payload,
    artifactHash: hashArtifactPayload(payload),
  };
}
