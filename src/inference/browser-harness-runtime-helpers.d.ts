export declare function parseReportTimestamp(
  rawTimestamp: string | number | Date | null | undefined,
  label?: string
): string | null;
export declare function resolveReportTimestamp(
  rawTimestamp: string | number | Date | null | undefined,
  label: string,
  fallbackTimestamp?: string | null
): string;
export declare function resolveRuntime(options: Record<string, unknown>): Record<string, unknown>;
export declare function cloneRuntimeConfig(runtimeConfig: Record<string, unknown> | null | undefined): Record<string, unknown> | null;
export declare function snapshotRuntimeState(): {
  runtimeConfig: Record<string, unknown> | null;
  activeKernelPath: unknown;
  activeKernelPathSource: string | null;
  activeKernelPathPolicy: unknown;
};
export declare function restoreRuntimeState(snapshot: Record<string, unknown> | null | undefined): void;
export declare function runWithRuntimeIsolationForSuite<T>(run: () => Promise<T>): Promise<T>;
export declare function sanitizeReportOutput(output: unknown): unknown;
export declare function loadRuntimeConfigFromRef(
  ref: string,
  context: Record<string, unknown>
): Promise<{ config: Record<string, unknown>; runtime: Record<string, unknown> }>;
export declare function loadRuntimeConfigFromUrl(
  url: string,
  options?: Record<string, unknown>
): Promise<{ config: Record<string, unknown>; runtime: Record<string, unknown> }>;
export declare function applyRuntimeConfigFromUrl(
  url: string,
  options?: Record<string, unknown>
): Promise<Record<string, unknown>>;
export declare function loadRuntimeProfile(
  profileId: string,
  options?: Record<string, unknown>
): Promise<{ config: Record<string, unknown>; runtime: Record<string, unknown> }>;
export declare function applyRuntimeProfile(
  profileId: string,
  options?: Record<string, unknown>
): Promise<Record<string, unknown>>;
export declare function normalizeRuntimeConfigChain(value: unknown): string[];
export declare function applyRuntimeForRun(
  run: Record<string, unknown>,
  options?: Record<string, unknown>
): Promise<void>;
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
