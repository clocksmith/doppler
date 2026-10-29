export declare function resolveReportTimestamp(
  rawTimestamp: string | number | Date | null | undefined,
  label: string,
  fallbackTimestamp?: string | null
): string;

export declare function sanitizeReportOutput(output: unknown): unknown;

export declare function normalizeManifest(manifest: Record<string, unknown>): {
  defaults: Record<string, unknown>;
  runs: Array<Record<string, unknown>>;
  reportModelId: string;
  report: Record<string, unknown> | null;
};

export declare function mergeRunDefaults(
  defaults: Record<string, unknown>,
  run: Record<string, unknown>
): Record<string, unknown>;

export declare function summarizeManifestRuns(results: Array<Record<string, unknown>>): {
  totalRuns: number;
  passedRuns: number;
  failedRuns: number;
  durationMs: number;
};
