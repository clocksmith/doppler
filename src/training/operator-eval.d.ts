export interface BleuResult {
  score: number;
  brevityPenalty: number;
  precisions: number[];
  hypothesisLength: number;
  referenceLength: number;
}

export interface ChrfResult {
  score: number;
  precision: number;
  recall: number;
}

export declare function computeBleuScore(
  hypotheses: string[],
  references: string[],
  options?: { maxOrder?: number }
): BleuResult;

export declare function computeChrfScore(
  hypotheses: string[],
  references: string[],
  options?: { maxOrder?: number; beta?: number }
): ChrfResult;

export declare function computeExactMatch(
  hypotheses: string[],
  references: string[]
): { score: number; matches: number; total: number };

export declare function computeAccuracy(
  labels: string[],
  predictions: string[]
): { score: number; matches: number; total: number };

export declare function computeEvalMetrics(
  evalKind: string,
  hypotheses: string[],
  references: string[],
  options?: Record<string, unknown>
): Record<string, unknown>;

export declare function loadEvalDataset(datasetPath: string): Promise<{
  absolutePath: string;
  rows: unknown[];
  raw: string;
}>;
