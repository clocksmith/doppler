export interface DemoState {
  phase: string;
  bootError: string | null;
  model: unknown;
  modelId: string | null;
  modelStatus: Record<string, string>;
  webgpuAvailable: boolean;
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
  modelBusy: boolean;
  settingsBusy: boolean;
  xrayEnabled: boolean;
  branchTree: import('../branches.js').BranchTree | null;
  branchPrompt: string;
  branchModelId: string | null;
  lastInspection: unknown;
  lastInferenceStats: Record<string, unknown> | null;
  lastRun: unknown;
  lastImportedReport: unknown;
}

export declare const state: DemoState;
