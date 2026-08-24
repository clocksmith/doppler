export interface SourceAssetDownloadOptions {
  signal?: AbortSignal | null;
  onProgress?: (receivedBytes: number) => void;
}

export interface SourceAssetDescriptor {
  path: string;
  size?: number | null;
  hash?: string | null;
  hashAlgorithm?: string | null;
}

export interface SourceAssetDownloadResult {
  source: 'http';
  path: string;
  bytes: number;
}

export declare function joinArtifactUrl(baseUrl: string, relativePath: string): string;
export declare function downloadSourceAsset(
  url: string | URL,
  asset: SourceAssetDescriptor,
  options?: SourceAssetDownloadOptions
): Promise<SourceAssetDownloadResult>;
export declare function downloadShard(
  baseUrl: string,
  shardIndex: number,
  shardInfo: Record<string, unknown>,
  options?: Record<string, unknown>
): Promise<Record<string, unknown>>;

