export interface BaselineMatchRule {
  match: string;
  matchType?: 'includes' | 'exact' | 'regex';
}

export interface BaselineMetricRange {
  min?: number;
  max?: number;
}

export interface BaselineEntry {
  id?: string;
  enabled?: boolean;
  model?: string | BaselineMatchRule;
  modelMatch?: string | BaselineMatchRule;
  gpu?: string | BaselineMatchRule;
  gpuMatch?: string | BaselineMatchRule;
  browser?: string | BaselineMatchRule;
  browserMatch?: string | BaselineMatchRule;
  metrics?: Record<string, BaselineMetricRange>;
}

export interface BaselineRegistry {
  schemaVersion?: number;
  baselines?: BaselineEntry[];
  _path?: string;
}

export interface BaselineViolation {
  metric: string;
  value: number;
  min: number | null;
  max: number | null;
}

export interface BaselineEvaluation {
  ok: boolean;
  violations: BaselineViolation[];
}

export function loadBaselineRegistry(path: string): Promise<BaselineRegistry>;
export function findBaselineForResult(result: unknown, registry: BaselineRegistry | null): BaselineEntry | null;
export function evaluateBaseline(result: unknown, baseline: BaselineEntry): BaselineEvaluation;
