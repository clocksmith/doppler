export interface EnsureModelCachedResult {
  cached: boolean;
  fromCache: boolean;
  cacheState: 'hit' | 'manifest-refresh' | 'imported' | 'error';
  modelId: string;
  error: string | null;
}

export interface CacheProgressEvent {
  stage: 'cache-hit' | 'cache-refresh' | 'cache-invalidate' | 'download-start' | 'downloading' | 'download-complete';
  modelId?: string;
  message?: string;
  percent?: number;
  totalBytes?: number;
  downloadedBytes?: number;
  totalShards?: number;
  completedShards?: number;
  speed?: number;
  speedFormatted?: string;
  totalFormatted?: string;
  downloadedFormatted?: string;
  eta?: string;
}

export declare function ensureModelCached(
  modelId: string,
  modelBaseUrl: string,
  onProgress?: ((progress: CacheProgressEvent) => void) | null
): Promise<EnsureModelCachedResult>;
