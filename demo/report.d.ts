export interface DemoReport {
  schema?: 'doppler.demo-report/v1';
  [key: string]: unknown;
}

export declare function buildReport(): DemoReport | null;
export declare function getReportOutput(report: unknown): string;
export declare function validateImportedReport(value: unknown): DemoReport;
export declare function importReportData(value: unknown): DemoReport;
export declare function exportReport(): void;
export declare function exportReferenceTranscript(): void;
export declare function setExportEnabled(enabled: boolean): void;
export declare function setTranscriptExportEnabled(enabled: boolean): void;
export declare function initReport(): void;
