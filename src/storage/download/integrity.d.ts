export interface DownloadSourceStats {
  cache: number;
  p2p: number;
  http: number;
  unknown: number;
}

export declare function buildManifestVersionSet(manifest: unknown): string;
export declare function createDefaultSourceStats(): DownloadSourceStats;
export declare function normalizeSourceStats(value: unknown): DownloadSourceStats;
export declare function isTokenizerJsonRequired(tokenizer: unknown): boolean;
export declare function getTokenizerModelPath(tokenizer: unknown): string | null;
export declare function fileExistsInStore(path: string): Promise<boolean>;
export declare function computeAssetHash(
  payload: ArrayBuffer | Uint8Array,
  algorithm?: string
): Promise<string>;
export declare function persistDownloadedShardIfNeeded(
  result: {
    source?: string;
    wrote?: boolean;
    buffer?: ArrayBuffer | null;
  } | null | undefined,
  shardIndex: number,
  options?: {
    writeShardFn?: (
      shardIndex: number,
      buffer: ArrayBuffer,
      options?: { verify?: boolean }
    ) => Promise<unknown>;
  }
): Promise<boolean>;

