import type { LoadedTrainingWorkload, DistillStagePlanEntry } from '../workloads.js';

export declare function runDistillationStage(options: {
  loadedWorkload: LoadedTrainingWorkload;
  stageEntry: DistillStagePlanEntry;
  layout: Record<string, string>;
  datasetPath?: string | null;
  stageAArtifact?: string | null;
  stageAArtifactHash?: string | null;
  legacyArtifactDir?: string | null;
  timestamp?: string | Date | null;
  parentArtifacts?: Array<Record<string, unknown>>;
}): Promise<{
  stageId: string;
  trainingStage: 'stage_a' | 'stage_b';
  metrics: Record<string, unknown>[];
  checkpointArtifacts: Array<Record<string, unknown>>;
  evalReports: Array<Record<string, unknown>>;
  bestReport: Record<string, unknown> | null;
  stageManifestPath: string;
  legacyArtifact: Record<string, unknown> | null;
  lastCheckpoint: Record<string, unknown> | null;
}>;

export declare function runDistillationStageA(options: {
  loadedWorkload: LoadedTrainingWorkload;
  stageEntry: DistillStagePlanEntry;
  layout: Record<string, string>;
  datasetPath?: string | null;
  stageAArtifact?: string | null;
  stageAArtifactHash?: string | null;
  legacyArtifactDir?: string | null;
  timestamp?: string | Date | null;
  parentArtifacts?: Array<Record<string, unknown>>;
}): Promise<{
  stageId: string;
  trainingStage: 'stage_a' | 'stage_b';
  metrics: Record<string, unknown>[];
  checkpointArtifacts: Array<Record<string, unknown>>;
  evalReports: Array<Record<string, unknown>>;
  bestReport: Record<string, unknown> | null;
  stageManifestPath: string;
  legacyArtifact: Record<string, unknown> | null;
  lastCheckpoint: Record<string, unknown> | null;
}>;
