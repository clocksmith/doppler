export interface AgentHeldoutEvalPolicy {
  suiteId?: string | null;
  categories?: string[];
  minPassRate?: number;
  requirePatchApplies?: boolean;
  requireNoHallucinatedFiles?: boolean;
  requireNoHallucinatedTools?: boolean;
  allowedFiles?: string[];
  allowedTools?: string[];
}

export interface AgentHeldoutEvalResult {
  schemaVersion: number;
  suiteId: string | null;
  totalRows: number;
  passedRows: number;
  failedRows: number;
  passRate: number;
  minPassRate: number;
  passed: boolean;
  requiredCategories: string[];
  missingCategories: string[];
  categorySummary: Record<string, {
    total: number;
    passed: number;
    passRate: number;
  }>;
  rows: Array<{
    id: string;
    categories: string[];
    passed: boolean;
    checks: Array<Record<string, unknown>>;
  }>;
}

export declare function extractFileReferences(text: string): string[];

export declare function extractToolReferences(text: string, knownTools?: readonly string[]): string[];

export declare function evaluateAgentHeldoutRows(
  datasetRows: Array<Record<string, unknown>>,
  candidateRows: Array<Record<string, unknown>>,
  options?: {
    policy?: AgentHeldoutEvalPolicy;
    patchStatuses?: Record<string, Record<string, unknown>>;
  }
): AgentHeldoutEvalResult;

export declare function summarizeAgentEvalReportRequirements(
  workload: Record<string, unknown>,
  reports: Array<Record<string, unknown>>
): {
  requiredCount: number;
  passedCount: number;
  failedCount: number;
  requirements: Array<{
    evalDatasetId: string;
    suiteId: string;
    minPassRate: number;
    requiredCategories: string[];
    reportCount: number;
    passed: boolean;
    passingReportPaths: string[];
  }>;
};
