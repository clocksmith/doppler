export interface QuickstartVendorBenchmark {
  transformersjs?: {
    repoId: string;
    dtype?: string;
  };
}

export interface QuickstartBenchmarkEvidence {
  status: string;
  localClaimLaneId: string;
  runtimeReport: string;
  compareResult: string;
  summarySvg: string;
}

export interface QuickstartRegistryEntry {
  modelId: string;
  sourceCheckpointId: string;
  weightPackId: string;
  manifestVariantId: string;
  artifactCompleteness: string;
  runtimePromotionState: string;
  weightsRefAllowed: boolean;
  aliases: string[];
  modes: string[];
  hf: {
    repoId: string;
    revision: string | null;
    path: string;
  } | null;
  vendorBenchmark?: QuickstartVendorBenchmark;
  benchmarkEvidence?: QuickstartBenchmarkEvidence;
}

export declare function listQuickstartModels(): Promise<Array<{
  modelId: string;
  sourceCheckpointId: string;
  weightPackId: string;
  manifestVariantId: string;
  artifactCompleteness: string;
  runtimePromotionState: string;
  weightsRefAllowed: boolean;
  aliases: string[];
  modes: string[];
  vendorBenchmark?: QuickstartVendorBenchmark;
  benchmarkEvidence?: QuickstartBenchmarkEvidence;
}>>;

export declare function resolveQuickstartModel(model: string): Promise<QuickstartRegistryEntry>;

export declare function buildQuickstartModelBaseUrl(
  entry: QuickstartRegistryEntry,
  options?: { cdnBasePath?: string }
): string;
