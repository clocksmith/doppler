export type ExpectedManifestHash = string | { hex?: string; hash?: string; digest?: string };
export declare function normalizeExpectedManifestHash(value: ExpectedManifestHash | null | undefined): string | null;
export declare function resolveManifestReference(baseUrl: string, options?: {
  manifestUrl?: string;
  expectedManifestHash?: ExpectedManifestHash;
}): { manifestUrl: string; expectedManifestHash: string | null };

export interface DownloadSourceStats {
  cache: number;
  p2p: number;
  http: number;
  unknown: number;
}

export declare function assertManifestHash(text: string, expectedHash: string | null): Promise<void>;

export declare function buildManifestVersionSet(manifest: unknown): string;
export declare function createDefaultSourceStats(): DownloadSourceStats;
export declare function normalizeSourceStats(value: unknown): DownloadSourceStats;
export declare function isTokenizerJsonRequired(tokenizer: unknown): boolean;
export declare function getTokenizerModelPath(tokenizer: unknown): string | null;
export declare function fileExistsInStore(path: string, readFile?: (path: string) => Promise<ArrayBuffer>): Promise<boolean>;
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
