import type { LoadedTrainingWorkload, DistillStagePlanEntry } from '../workloads.js';

export declare function resolveInternalDistillStage(
  stageEntry: DistillStagePlanEntry | Record<string, unknown>
): 'stage-a' | 'stage-b';

export declare function buildDistillationTrainingConfigFromWorkload(
  loadedWorkload: LoadedTrainingWorkload,
  stageEntry: DistillStagePlanEntry | Record<string, unknown>,
  options?: {
    datasetPath?: string | null;
    artifactDir?: string | null;
    stageAArtifact?: string | null;
    stageAArtifactHash?: string | null;
  }
): {
  internalStage: 'stage-a' | 'stage-b';
  trainingConfig: Record<string, unknown>;
  trainingConfigHash: string;
};
