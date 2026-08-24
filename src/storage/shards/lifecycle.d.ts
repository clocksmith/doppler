export interface ShardWriterOptions {
  append?: boolean;
  expectedOffset?: number | null;
}

export interface NormalizedShardWriterOptions {
  append: boolean;
  expectedOffset: number | null;
}

export declare function normalizeShardWriterOptions(
  options?: ShardWriterOptions
): NormalizedShardWriterOptions;

export declare function createStorageWriteStream(
  storageBackend: {
    createWriteStream(filename: string, options: NormalizedShardWriterOptions): unknown;
  },
  filename: string,
  options?: ShardWriterOptions,
  onCreate?: (() => void) | null
): unknown;
