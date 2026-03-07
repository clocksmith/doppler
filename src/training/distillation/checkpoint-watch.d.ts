import type { LoadedTrainingWorkload } from '../workloads.js';

export declare function watchDistillationCheckpoints(options: {
  loadedWorkload: LoadedTrainingWorkload;
  layout: Record<string, string>;
  checkpointsDir?: string | null;
  manifestPath?: string | null;
  pollIntervalMs?: number | null;
  stopWhenIdle?: boolean;
}): Promise<{ ok: true; processedCount: number; manifestPath: string }>;
