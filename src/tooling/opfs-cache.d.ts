export interface EnsureModelCachedResult {
  cached: boolean;
  fromCache: boolean;
  modelId: string;
  error: string | null;
}

export declare function ensureModelCached(
  modelId: string,
  modelBaseUrl: string
): Promise<EnsureModelCachedResult>;
