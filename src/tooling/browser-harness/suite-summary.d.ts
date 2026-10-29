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
