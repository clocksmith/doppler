import type { LoadedTrainingWorkload } from './workloads.js';

export declare function runLoraPipeline(options: {
  loadedWorkload: LoadedTrainingWorkload;
  runRoot?: string | null;
  timestamp?: string | Date | null;
}): Promise<Record<string, unknown>>;

export declare function evaluateLoraCheckpoint(options: {
  loadedWorkload: LoadedTrainingWorkload;
  checkpointPath: string;
  checkpointId?: string | null;
  checkpointStep?: number | null;
  layout?: Record<string, string> | null;
}): Promise<Record<string, unknown>[]>;

export declare function exportLoraCheckpoint(options: {
  loadedWorkload: LoadedTrainingWorkload;
  checkpointPath: string;
  checkpointId?: string | null;
  checkpointStep?: number | null;
  layout?: Record<string, string> | null;
  exportsDir?: string | null;
  datasetHash?: string | null;
}): Promise<Record<string, unknown>>;

export declare function watchLoraCheckpoints(options: {
  loadedWorkload: LoadedTrainingWorkload;
  runRoot: string;
  pollIntervalMs?: number | null;
  stopWhenIdle?: boolean;
}): Promise<{ ok: true; processedCount: number; manifestPath: string }>;

export declare function compareLoraRun(options: {
  runRoot: string;
}): Promise<Record<string, unknown>>;

export declare function qualityGateLoraRun(options: {
  runRoot: string;
}): Promise<Record<string, unknown>>;
