export declare function watchFinalizedCheckpoints(options: {
  checkpointsDir: string;
  manifestPath: string;
  pollIntervalMs?: number | null;
  stopWhenIdle?: boolean;
  signal?: AbortSignal | null;
  onCheckpoint: (markerPath: string) => Promise<void> | void;
}): Promise<{ ok: true; processedCount: number; manifestPath: string; aborted?: boolean }>;
