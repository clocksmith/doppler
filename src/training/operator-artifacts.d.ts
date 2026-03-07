import type { LoadedTrainingWorkload, TrainingWorkloadPack } from './workloads.js';

export interface TrainingRunLayout {
  runRoot: string;
  logs: string;
  checkpoints: string;
  eval: string;
  scoreboard: string;
  exports: string;
  compare: string;
  qualityGate: string;
}

export declare function normalizeTrainingTimestamp(value?: string | Date | null): string;

export declare function createTrainingRunLayout(options: {
  kind: string;
  workloadId: string;
  timestamp?: string | Date | null;
}): Promise<TrainingRunLayout>;

export declare function writeJsonArtifact(
  filePath: string,
  payload: Record<string, unknown>
): Promise<{ path: string; sha256: string; relativePath: string }>;

export declare function writeNdjsonRow(
  filePath: string,
  row: Record<string, unknown>
): Promise<{ path: string; sha256: string; relativePath: string }>;

export declare function writeWorkloadLock(
  layout: TrainingRunLayout,
  loadedWorkload: LoadedTrainingWorkload
): Promise<{ path: string; sha256: string; relativePath: string }>;

export declare function writeRunContract(
  layout: TrainingRunLayout,
  payload: Record<string, unknown>
): Promise<{ path: string; sha256: string; relativePath: string }>;

export declare function buildArtifactBase(options: {
  artifactType: string;
  reportId: string;
  workload: TrainingWorkloadPack;
  workloadPath: string;
  workloadSha256: string;
  datasetPath: string | null;
  datasetHash: string | null;
  baseModelId: string | null;
  teacherModelId?: string | null;
  studentModelId?: string | null;
  stage?: string | null;
  checkpointStep?: number | null;
  parentArtifacts?: Array<Record<string, unknown>>;
  runtime?: string;
  surface?: string;
  claimBoundary: string;
  configHash: string;
}): Record<string, unknown>;

export declare function hashArtifactPayload(payload: Record<string, unknown>): string;
