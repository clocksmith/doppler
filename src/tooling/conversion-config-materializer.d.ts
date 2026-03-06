export declare function extractTensorEntriesFromManifest(
  manifest: Record<string, unknown>
): Array<{
  name: string;
  dtype: unknown;
  shape: unknown;
  role: unknown;
  layout: unknown;
}>;

export declare function resolveMaterializedManifestFromConversionConfig(
  conversionConfigInput: Record<string, unknown>,
  manifest: Record<string, unknown>
): {
  modelId: string;
  modelType: string;
  architecture: Record<string, unknown> | null;
  inference: Record<string, unknown> | null;
};

export declare function inferConversionConfigModelId(
  configPath: string,
  conversionConfigInput: Record<string, unknown>
): string;
