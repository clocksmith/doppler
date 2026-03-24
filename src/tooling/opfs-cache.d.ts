export interface EnsureModelCachedResult {
  cached: boolean;
  fromCache: boolean;
  cacheState: 'hit' | 'manifest-refresh' | 'imported' | 'error';
  modelId: string;
  error: string | null;
}

export declare function ensureModelCached(
  modelId: string,
  modelBaseUrl: string
): Promise<EnsureModelCachedResult>;
