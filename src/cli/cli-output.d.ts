import type { DopplerComparisonFingerprint } from '../client/inspection.js';

export function toSummary(result: any): string;
export function isFailedVerificationResult(request: any, result: any): boolean;
export function formatNumber(value: any, digits?: number): string;
export function formatMs(value: any): string;
export function compactTimestamp(): string;
export function saveBenchResult(result: any, saveDir: any): Promise<any>;
export function loadBaseline(comparePath: string, saveDir: string): Promise<Record<string, unknown>>;
export interface ComparableBenchmarkResult {
    modelId: string;
    suite: 'bench';
    workload: string;
    failed: 0;
    comparisonFingerprint: DopplerComparisonFingerprint;
    env: Record<string, unknown>;
    cacheMode: string;
    loadMode: string;
    metrics: Record<string, unknown> & {
        warmupRuns: number;
        timedRuns: number;
        seed: number | null;
        maxTokens: number;
        prompt: string;
    };
}
export function compareBenchResults(current: ComparableBenchmarkResult, baseline: ComparableBenchmarkResult): {
    regressions: string[];
    improvements: string[];
};
export function printManifestSummary(results: any): void;
export function printDeviceInfo(result: any): void;
export function printConvertContractSummary(result: any): void;
export function printConvertReportSummary(result: any): void;
export function printMetricsSummary(result: any): void;
