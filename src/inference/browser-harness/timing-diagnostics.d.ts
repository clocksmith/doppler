export declare function toTimingNumber(value: unknown, fallback?: number | null): number | null;
export declare function safeToFixed(value: unknown, fallback?: number | null, digits?: number): number | null;
export declare function sampleTimingNumber(
  stats: Record<string, unknown> | null | undefined,
  key: string,
  fallback?: number | null
): number | null;
export declare function buildCanonicalTiming(overrides?: Record<string, unknown>): Record<string, unknown>;
export declare function buildLoadTimingDiagnostics(
  modelLoadMs: unknown,
  loadTiming?: Record<string, unknown> | null,
  pipelineLoadTiming?: Record<string, unknown> | null
): Record<string, unknown> | null;
export declare function buildTimingDiagnostics(timing?: Record<string, unknown>, options?: Record<string, unknown>): Record<string, unknown>;
export declare function buildFirstLoadComposition(
  fields?: Record<string, unknown>
): Record<string, number | null>;
