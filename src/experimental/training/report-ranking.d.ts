export interface TrainingReport {
  [key: string]: unknown;
  metrics?: Record<string, number | { score?: number } | null>;
  primaryMetric?: string;
  qualityClaim?: { improved?: boolean; requireImprovement?: boolean } | null;
}
export interface ComparisonWorkload {
  selectionMetric?: string;
  selectionGoal?: string;
  evalDatasets?: { id: string; quality?: { requireImprovement?: boolean } }[];
}
export declare function resolveComparableReportMetric(report: TrainingReport | null | undefined, metric: string): number | null;
export declare function sortTrainingReports<T extends TrainingReport>(reports: T[], workload?: ComparisonWorkload | null): T[];
export declare function collectRequiredImprovementEvalIds(workload: ComparisonWorkload | null | undefined): string[];
export declare function summarizeQualityClaims(reports: TrainingReport[]): { count: number; improvedCount: number; requiredCount: number; failedRequiredCount: number };
