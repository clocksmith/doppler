export interface EnsureModelCachedResult {
  cached: boolean;
  fromCache: boolean;
  cacheState: 'hit' | 'verified-hit' | 'manifest-refresh' | 'imported' | 'error';
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

export interface CachedModelSourceResult extends EnsureModelCachedResult {
  manifest: Record<string, unknown>;
  manifestText: string;
  manifestHash: string;
  storageContext: Record<string, unknown>;
  storageBackend: 'opfs';
  totalBytes: number;
}

export interface EnsureModelCachedSourceOptions {
  expectedManifestHash?: string | { hex?: string; hash?: string; digest?: string };
}

export declare function ensureModelCached(
  modelId: string,
  modelBaseUrl: string,
  onProgress?: ((progress: CacheProgressEvent) => void) | null
): Promise<EnsureModelCachedResult>;

export declare function ensureModelCachedSource(
  modelId: string,
  modelBaseUrl: string,
  onProgress?: ((progress: CacheProgressEvent) => void) | null,
  options?: EnsureModelCachedSourceOptions
): Promise<CachedModelSourceResult>;
