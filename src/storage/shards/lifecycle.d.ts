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
