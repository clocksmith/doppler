export function toSummary(result: any): string;
export function formatNumber(value: any, digits?: number): string;
export function formatMs(value: any): string;
export function compactTimestamp(): string;
export function saveBenchResult(result: any, saveDir: any): Promise<any>;
export function loadBaseline(comparePath: any, saveDir: any): Promise<any>;
export function compareBenchResults(current: any, baseline: any): {
    regressions: string[];
    improvements: string[];
};
export function printManifestSummary(results: any): void;
export function printDeviceInfo(result: any): void;
export function printConvertContractSummary(result: any): void;
export function printConvertReportSummary(result: any): void;
export function printMetricsSummary(result: any): void;
