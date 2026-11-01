import type { DopplerComparisonFingerprint } from '../../client/inspection.js';
import type { PipelineStats } from '../../inference/pipelines/text/types.js';

export interface BenchmarkEnvironment {
  library: string;
  runtime: 'node' | 'browser';
  device: string;
  browserUserAgent: string | null;
  browserPlatform: string | null;
  browserLanguage: string | null;
  browserVendor: string | null;
}
export interface BenchmarkComparison {
  comparisonFingerprint: DopplerComparisonFingerprint | null;
  comparisonFingerprintUnavailableReason: string | null;
}
export interface BenchmarkComparisonHarness {
  manifestHash?: string | null;
  manifest?: { modelId?: string; tokenizer?: unknown };
  pipeline: { getStats?: () => PipelineStats };
}
export interface BenchmarkComparisonRun {
  initialExecutionIdentity?: { digest: string } | null;
  generationConfig?: Record<string, unknown>;
  promptTokenIds?: number[] | null;
}
export interface BenchmarkObservationConfig extends Record<string, unknown> {
  shared?: {
    harness?: { mode?: string };
    debug?: { probes?: unknown[] };
  };
}
export type CaptureBenchmarkComparison = (
  harness: BenchmarkComparisonHarness,
  run: BenchmarkComparisonRun,
  runtimeConfig: BenchmarkObservationConfig,
  runOptions: Record<string, unknown>,
  deviceInfo: (Record<string, unknown> & { adapterInfo?: Record<string, unknown> }) | null
) => BenchmarkComparison;
export interface BenchmarkComparisonCapture {
  result: BenchmarkComparison;
  capture(run: BenchmarkComparisonRun): void;
}
export type CreateBenchmarkComparisonCapture = (
  harness: BenchmarkComparisonHarness,
  runtimeConfig: BenchmarkObservationConfig,
  runOptions: Record<string, unknown>,
  deviceInfo: (Record<string, unknown> & { adapterInfo?: Record<string, unknown> }) | null
) => BenchmarkComparisonCapture;
export declare const createBenchmarkComparisonCapture: CreateBenchmarkComparisonCapture;
export declare function resolveBenchmarkEnvironment(): BenchmarkEnvironment;
