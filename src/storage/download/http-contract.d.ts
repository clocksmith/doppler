export const DISTRIBUTION_SOURCE_HTTP: 'http';
export function normalizeRequiredInteger(value: unknown, label: string, options?: { allowZero?: boolean; fallback?: number | null }): number;
export function createShardSizeMismatchError(message: string, details?: Record<string, unknown>): Error;
export function normalizeManifestVersionSet(value: unknown): string | null;
export function assertExpectedHash(resultHash: string | null, expectedHash: string | null, shardIndex: number): void;
export function assertExpectedSize(bytes: number, expectedSize: number | null, shardIndex: number): void;
export function assertExpectedManifestVersionSet(result: string | null, expected: string | null, index: number, source: string): void;
export function createAbortError(label?: string): Error;
export function createSourceCounter(): Record<'cache' | 'p2p' | 'http', number>;
export function createLatencySummary(durations: number[]): { count: number; min: number | null; max: number | null; avg: number | null };
export function createDeliveryMetrics(order: string[], result: import('../distribution-transport.js').DownloadShardResult,
  attempts: { status: string; source: string; durationMs?: number; code?: string; writeDurationMs?: number }[], totalDurationMs: number
): import('../distribution-transport.js').ShardDeliveryMetrics;
