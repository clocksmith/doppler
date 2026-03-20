export declare function buildSuiteSummary(
  suiteName: string,
  results: Array<Record<string, unknown>>,
  startTimeMs: number
): {
  suite: string;
  passed: number;
  failed: number;
  skipped: number;
  duration: number;
  results: Array<Record<string, unknown>>;
};
export declare function normalizeCacheMode(value: unknown): 'cold' | 'warm';
export declare function normalizeLoadMode(value: unknown, hasModelUrl: boolean, modelUrl?: string | null): 'opfs' | 'http' | 'memory' | 'file';
export declare function normalizeWorkloadType(value: unknown): string | null;
export declare function safeStatsValue(value: unknown): number;
export declare function calculateRatePerSecond(count: unknown, durationMs: unknown): number;
export declare function buildDiffusionPerformanceArtifact(options: Record<string, unknown>): Record<string, unknown>;
export declare function assertDiffusionPerformanceArtifact(metrics: Record<string, unknown>, contextLabel?: string): void;
export declare function toTimingNumber(value: unknown, fallback?: number | null): number | null;
export declare function safeToFixed(value: unknown, fallback?: number | null, digits?: number): number | null;
export declare function sampleTimingNumber(
  stats: Record<string, unknown> | null | undefined,
  key: string,
  fallback?: number | null
): number | null;
export declare function buildCanonicalTiming(overrides?: Record<string, unknown>): Record<string, unknown>;
export declare function buildTimingDiagnostics(timing?: Record<string, unknown>, options?: Record<string, unknown>): Record<string, unknown>;
