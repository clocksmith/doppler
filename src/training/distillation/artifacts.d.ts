import type { LoadedTrainingWorkload } from '../workloads.js';

export declare function createDistillationRunArtifacts(options: {
  loadedWorkload: LoadedTrainingWorkload;
  runRoot?: string | null;
  timestamp?: string | Date | null;
}): Promise<{
  layout: {
    runRoot: string;
    logs: string;
    checkpoints: string;
    eval: string;
    scoreboard: string;
    exports: string;
    compare: string;
    qualityGate: string;
  };
  runContract: { path: string; sha256: string; relativePath: string };
  workloadLock: { path: string; sha256: string; relativePath: string };
  runContractPayload: Record<string, unknown>;
}>;

export declare function writeDistillStageManifest(
  layout: Record<string, string>,
  payload: Record<string, unknown>
): Promise<{ path: string; sha256: string; relativePath: string }>;

export declare function writeDistillCheckpointMetadata(
  layout: Record<string, string>,
  stageId: string,
  checkpointId: string,
  payload: Record<string, unknown>
): Promise<{ path: string; sha256: string; relativePath: string }>;

export declare function writeDistillCheckpointComplete(
  layout: Record<string, string>,
  stageId: string,
  checkpointId: string,
  payload: Record<string, unknown>
): Promise<{ path: string; sha256: string; relativePath: string }>;

export declare function writeDistillEvalReport(
  layout: Record<string, string>,
  payload: Record<string, unknown>
): Promise<{ path: string; sha256: string; relativePath: string }>;

export declare function writeDistillCompareReport(
  layout: Record<string, string>,
  payload: Record<string, unknown>
): Promise<{ path: string; sha256: string; relativePath: string }>;

export declare function writeDistillQualityGateReport(
  layout: Record<string, string>,
  payload: Record<string, unknown>
): Promise<{ path: string; sha256: string; relativePath: string }>;

export declare function buildDistillArtifactBase(
  loadedWorkload: LoadedTrainingWorkload,
  options: {
    prefix?: string;
    artifactType: string;
    datasetPath?: string | null;
    datasetHash?: string | null;
    stage?: string | null;
    checkpointStep?: number | null;
    parentArtifacts?: Array<Record<string, unknown>>;
    runtime?: string;
    surface?: string;
    configHash?: string | null;
  }
): Record<string, unknown>;
