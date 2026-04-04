export interface TranslateCompareMetrics {
  modelLoadMs: number | null;
  ttftMs: number | null;
  totalMs: number | null;
  decodeTokensPerSec: number | null;
  sizeBytes: number | null;
  deviceLabel: string | null;
  metaLabel: string | null;
}

export interface TranslateCompareLaneState {
  engine: 'doppler' | 'transformersjs';
  modelId: string | null;
  tjsModelId: string;
  tjsDtype: string;
  label: string;
  status: string;
  statusTone: string;
  output: string;
  metrics: TranslateCompareMetrics | null;
  error: unknown;
  pipeline: any;
  pipelineModelId: string | null;
  tjsGenerator: any;
  tjsGeneratorKey: string | null;
}

export interface TranslateCompareProfile {
  dopplerModelId: string;
  defaultTjsModelId: string;
  defaultKernelPath: string;
  modelBaseDir: string;
  defaultDopplerSurface: string;
}

export interface TranslateCompareEvidenceModel {
  label: string;
  modelId: string | null;
  bleu: number | null;
  chrf: number | null;
  sizeBytes: number | null;
}

export interface TranslateCompareEvidenceReceipt {
  label?: string;
  href?: string;
  [key: string]: unknown;
}

export interface TranslateCompareEvidence {
  updatedAt: string | null;
  summary: string;
  caution: string;
  teacher: TranslateCompareEvidenceModel;
  student: TranslateCompareEvidenceModel;
  receipts: TranslateCompareEvidenceReceipt[];
}

export interface TranslateCompareHistoryLane {
  engine: string;
  modelId: string;
  tjsModelId: string;
  label: string;
  status: string;
  output: string;
  metrics: TranslateCompareMetrics | null;
  error: unknown;
}

export interface TranslateCompareArtifactLane {
  engine: string;
  modelId: string;
  modelLabel: string;
  tjsModelId: string;
  roleLabel: string;
  status: string;
  output: string;
  metrics: TranslateCompareMetrics | null;
  error: unknown;
}

export interface TranslateCompareArtifact {
  schemaVersion: number;
  kind: string;
  artifactId: string;
  createdAt: string;
  shareUrl: string | null;
  request: {
    prompt: string;
    sourceCode: string;
    sourceName: string;
    targetCode: string;
    targetName: string;
    options: Record<string, unknown>;
    layoutId: string;
  };
  environment: Record<string, unknown>;
  evidence: {
    updatedAt: string | null;
    summary: string;
    receipts: TranslateCompareEvidenceReceipt[];
  };
  summary: Record<string, unknown>;
  lanes: {
    left: TranslateCompareArtifactLane;
    right: TranslateCompareArtifactLane;
  };
}

export interface TranslateCompareHistoryEntry {
  id: string;
  createdAt: string;
  sourceCode: string;
  targetCode: string;
  prompt: string;
  layoutId: string;
  artifact: TranslateCompareArtifact | null;
  lanes: {
    left: TranslateCompareHistoryLane;
    right: TranslateCompareHistoryLane;
  };
}

export interface State {
  runtimeOverride: any;
  runtimeOverrideBase: any;
  runtimeOverrideLabel: string | null;
  diagnosticsRuntimeConfig: any;
  diagnosticsRuntimeProfileId: string | null;
  diagnosticsSelections: Record<string, any>;
  lastDiagnosticsSuite: string | null;
  lastDiffusionRequest: any;
  lastReport: any;
  lastReportInfo: any;
  lastMetrics: any;
  lastInferenceStats: any;
  lastMemoryStats: any;
  activePipeline: any;
  activePipelineModelId: string | null;
  activeModelId: string | null;
  registeredModelIds: string[];
  modelTypeCache: Record<string, string>;
  modelAvailability: {
    total: number;
    run: number;
    translate: number;
    embedding: number;
    diffusion: number;
  };
  modeModelId: {
    run: string | null;
    translate: string | null;
    embedding: string | null;
    diffusion: string | null;
  };
  runAbortController: AbortController | null;
  runGenerating: boolean;
  runPrefilling: boolean;
  runLoading: boolean;
  compareEnabled: boolean;
  compareLayoutId: string;
  compareLoading: boolean;
  compareGenerating: boolean;
  compareHistory: TranslateCompareHistoryEntry[];
  compareHistoryFilter: string;
  compareProfiles: TranslateCompareProfile[];
  compareEvidence: TranslateCompareEvidence | null;
  compareDeviceLabel: string | null;
  activeCompareSmokeSampleId: string | null;
  lastCompareArtifact: TranslateCompareArtifact | null;
  compareLanes: {
    left: TranslateCompareLaneState;
    right: TranslateCompareLaneState;
  };
  diffusionGenerating: boolean;
  diffusionLoading: boolean;
  convertActive: boolean;
  downloadActive: boolean;
  surface: string;
  uiTask: string;
  uiMode: string;
  lastPrimaryMode: string;
  lastTaskMode: Record<string, string>;
  runLog: any[];
  runCounter: number;
  embeddingDemoDocuments: any[];
  storageUsageBytes: number;
  storageQuotaBytes: number;
  storageInspectorScanning: boolean;
  storageInspectorLastScan: number;
  gpuMaxBytes: number;
  systemMemoryBytes: number;
  uiIntervalId: any;
  lastStorageRefresh: number;
  activeDownloadId: string | null;
  quickModelCatalog: any[];
  quickModelCatalogError: string | null;
  quickModelCatalogLoading: boolean;
  kernelPathBuilderIndex: any;
  kernelPathBuilderLoading: boolean;
  kernelPathBuilderError: string | null;
  kernelPathBuilderModelId: string | null;
  kernelPathBuilderOverlayReport: any;
  kernelPathBuilderOverlaySource: string | null;
  quickModelActionModelId: string | null;
  quickModelStorageIds: string[];
  downloadProgress: any;
}

export declare const state: State;
