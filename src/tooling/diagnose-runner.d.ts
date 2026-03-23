import type { ToolingCommandRequest } from './command-api.js';

export interface DiagnoseRunSummary {
  label: 'baseline' | 'observed' | string;
  provider: string;
  modelId: string | null;
  output: string | null;
  timing: Record<string, unknown> | null;
  deviceInfo: Record<string, unknown> | null;
  metrics: Record<string, unknown>;
  operatorDiagnostics: Record<string, unknown> | null;
  reportInfo: Record<string, unknown> | null;
  artifacts: Record<string, unknown> | null;
}

export interface DiagnoseCommandResult {
  mode: 'operator_diff';
  baselineProvider: string;
  observedProvider: string;
  baseline: DiagnoseRunSummary;
  observed: DiagnoseRunSummary;
  divergence: Record<string, unknown>;
}

export declare function runDiagnoseCommand(
  request: ToolingCommandRequest,
  options?: Record<string, unknown>
): Promise<DiagnoseCommandResult>;
