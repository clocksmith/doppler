export interface GepaCandidate {
  id: string;
  content: string;
  scores: Record<string, number>;
  metrics: Record<string, unknown>;
  traces: unknown[];
  generation: number | null;
  rank: number | null;
  dominatedBy: number | null;
  crowdingDistance: number | null;
  targetType: string | null;
  sourcePolicyId: string;
}

export interface GepaFrontierSummary {
  candidateIds: string[];
  sourcePolicyIds: string[];
  objectiveNames: string[];
}

export function normalizeGepaCandidate(record: unknown, index?: number, options?: Record<string, unknown>): GepaCandidate;
export function parseGepaFrontier(text: string, options?: Record<string, unknown>): {
  candidates: GepaCandidate[];
  lineage: GepaFrontierSummary;
};
export function loadGepaFrontier(frontierPath: string, options?: Record<string, unknown>): Promise<{
  absolutePath: string;
  raw: string;
  candidates: GepaCandidate[];
  lineage: GepaFrontierSummary;
}>;
export function buildTeacherTracesFromGepaFrontier(candidates: GepaCandidate[], options?: Record<string, unknown>): Array<Record<string, unknown>>;
export function serializeGepaTeacherTraces(rows: unknown[]): string;
export function writeGepaTeacherTraces(frontierPath: string, outputPath: string, options?: Record<string, unknown>): Promise<{
  inputPath: string;
  outputPath: string;
  candidateCount: number;
  rowCount: number;
  lineage: GepaFrontierSummary;
}>;
export function summarizeGepaFrontier(candidates: GepaCandidate[]): GepaFrontierSummary;
