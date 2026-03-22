export interface KernelPathBuilderConfigEntry {
  configPath: string;
  rawConfig: Record<string, unknown>;
}

export interface KernelPathBuilderManifestEntry {
  manifestPath: string;
  manifest: Record<string, unknown>;
  modelId?: string;
}

export interface KernelPathBuilderRegistryEntry {
  id: string;
  status?: string;
  statusReason?: string;
  notes?: string;
  path: Record<string, unknown>;
}

export interface KernelPathBuilderIndexOptions {
  configEntries?: KernelPathBuilderConfigEntry[];
  manifestEntries?: KernelPathBuilderManifestEntry[];
  registryEntries?: KernelPathBuilderRegistryEntry[];
}

export interface KernelPathBuilderIndexPayload {
  schemaVersion: number;
  source: string;
  stats: Record<string, number>;
  skipped: Array<Record<string, unknown>>;
  kernelPaths: Array<Record<string, unknown>>;
  reverseIndexes: Record<string, Record<string, string[]>>;
  models: Array<Record<string, unknown>>;
}

export declare function buildKernelPathBuilderIndex(
  options?: KernelPathBuilderIndexOptions
): KernelPathBuilderIndexPayload;

export declare function buildKernelPathBuilderProposals(
  indexPayload: KernelPathBuilderIndexPayload
): Record<string, unknown>;

export declare function renderKernelPathBuilderReportMarkdown(
  indexPayload: KernelPathBuilderIndexPayload,
  proposalsPayload?: Record<string, unknown> | null
): string;

export declare function aggregateTopDecodeTimers(
  decodeProfileSteps: Array<Record<string, unknown>> | null | undefined,
  limit?: number
): Array<{ label: string; totalMs: number }>;

export declare function buildKernelPathBuilderRuntimeOverlay(
  model: Record<string, unknown> | null | undefined,
  report: Record<string, unknown> | null | undefined
): Record<string, unknown> | null;

export declare function buildKernelPathBuilderArtifacts(
  options?: KernelPathBuilderIndexOptions
): {
  index: KernelPathBuilderIndexPayload;
  proposals: Record<string, unknown>;
  reportMarkdown: string;
};
