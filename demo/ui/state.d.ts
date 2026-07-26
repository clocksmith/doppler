export interface DemoState {
  phase: string;
  bootError: string | null;
  model: unknown;
  modelId: string | null;
  modelStatus: Record<string, string>;
  quickModelCatalog: Array<Record<string, unknown>>;
  downloadProgress: Record<string, unknown> | null;
  generating: boolean;
  prefilling: boolean;
  abortController: AbortController | null;
  conversationHistory: Array<{ role: string; content: string }>;
  conversationModelId: string | null;
  settings: Record<string, unknown>;
  preset: string;
  wordQualityEnabled: boolean;
  liveTokSec: boolean;
  xrayEnabled: boolean;
  lastInspection: unknown;
  lastInferenceStats: Record<string, unknown> | null;
  lastRun: unknown;
  lastImportedReport: unknown;
}

export declare const state: DemoState;
