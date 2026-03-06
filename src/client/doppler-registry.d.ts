export interface QuickstartRegistryEntry {
  modelId: string;
  aliases: string[];
  modes: string[];
  hf: {
    repoId: string;
    revision: string | null;
    path: string;
  } | null;
}

export declare function listQuickstartModels(): Promise<Array<{
  modelId: string;
  aliases: string[];
  modes: string[];
}>>;

export declare function resolveQuickstartModel(model: string): Promise<QuickstartRegistryEntry>;

export declare function buildQuickstartModelBaseUrl(
  entry: QuickstartRegistryEntry,
  options?: { cdnBasePath?: string }
): string;
