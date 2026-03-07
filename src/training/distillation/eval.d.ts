import type { LoadedTrainingWorkload } from '../workloads.js';

export declare function evaluateDistillationModel(options: {
  loadedWorkload: LoadedTrainingWorkload;
  layout?: Record<string, string> | null;
  stageId: string;
  checkpointId: string;
  checkpointStep: number | null;
  checkpointPath?: string | null;
  distillRuntime: Record<string, unknown>;
  model: Record<string, unknown>;
  evalDatasetId?: string | null;
  configHash?: string | null;
  parentArtifacts?: Array<Record<string, unknown>>;
}): Promise<Record<string, unknown>[]>;

export declare function evaluateDistillationCheckpoint(options: {
  loadedWorkload: LoadedTrainingWorkload;
  checkpointPath: string;
  checkpointId?: string | null;
  checkpointStep?: number | null;
  stageId?: string | null;
  layout?: Record<string, string> | null;
  datasetPath?: string | null;
  stageAArtifact?: string | null;
  stageAArtifactHash?: string | null;
  evalDatasetId?: string | null;
  parentArtifacts?: Array<Record<string, unknown>>;
}): Promise<Record<string, unknown>[]>;

export declare function readDistillCheckpointMarker(markerPath: string): Promise<{
  absolutePath: string;
  marker: Record<string, unknown>;
}>;
