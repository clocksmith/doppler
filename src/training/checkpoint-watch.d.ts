export declare function watchFinalizedCheckpoints(options: {
  checkpointsDir: string;
  manifestPath: string;
  pollIntervalMs?: number | null;
  stopWhenIdle?: boolean;
  onCheckpoint: (markerPath: string) => Promise<void> | void;
}): Promise<{ ok: true; processedCount: number; manifestPath: string }>;
