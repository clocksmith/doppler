/**
 * reports.ts - Diagnostics report storage
 *
 * @module storage/reports
 */

export interface SavedReportInfo {
  backend: 'opfs' | 'indexeddb';
  path: string;
}

export interface SaveReportOptions {
  timestamp?: string | Date;
}

export declare function saveReport(
  modelId: string,
  report: Record<string, unknown>,
  options?: SaveReportOptions
): Promise<SavedReportInfo>;
