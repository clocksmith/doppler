export interface TeacherTraceRow {
  id: string;
  prompt: string;
  completion: string;
  promptField: string;
  completionField: string;
  teacherModelId: string;
  studentBaseModelId: string | null;
  domain: string | null;
  taskKind: string | null;
  policyId: string | null;
  sourcePolicyId: string | null;
  gepaCandidateId: string | null;
  sourceFiles: string[] | null;
  generationParams: Record<string, unknown> | null;
  license: string | null;
  provenance: Record<string, unknown> | null;
}

export interface TeacherTraceLineageSummary {
  teacherModelIds: string[];
  studentBaseModelIds: string[];
  taskKinds: string[];
  policyIds: string[];
  sourcePolicyIds: string[];
  gepaCandidateIds: string[];
}

export interface TeacherTraceDataset {
  sourceLabel: string;
  rowCount: number;
  rows: TeacherTraceRow[];
  textPairs: Array<Record<string, string | null>>;
  lineage: TeacherTraceLineageSummary;
}

export function normalizeTeacherTrace(record: unknown, index?: number, options?: Record<string, unknown>): TeacherTraceRow;
export function mapTeacherTraces(records: unknown[], options?: Record<string, unknown>): TeacherTraceRow[];
export function parseTeacherTraceDataset(text: string, options?: Record<string, unknown>): TeacherTraceDataset;
export function loadTeacherTraceDataset(datasetPath: string, options?: Record<string, unknown>): Promise<TeacherTraceDataset & {
  absolutePath: string;
  raw: string;
}>;
export function serializeTeacherTraceTextPairs(rows: unknown[]): string;
export function writeTeacherTraceTextPairs(inputPath: string, outputPath: string, options?: Record<string, unknown>): Promise<{
  inputPath: string;
  outputPath: string;
  rowCount: number;
  lineage: TeacherTraceLineageSummary;
}>;
export function summarizeTeacherTraceLineage(rows: TeacherTraceRow[]): TeacherTraceLineageSummary;
