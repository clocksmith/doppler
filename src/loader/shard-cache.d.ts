import type { RDRRManifest } from '../storage/rdrr-format.js';
import type { ShardCacheConfigSchema } from '../config/schema/loading.schema.js';
import type {
  CustomShardLoader,
  CustomShardRangeLoader,
  CustomShardStreamLoader,
  CustomShardStreamOptions,
  ShardLoadOptions,
  ShardSourceInfo,
} from './loader-types.js';

export interface ShardCacheConfig {
  maxEntries: number;
  customLoader?: CustomShardLoader | null;
  customRangeLoader?: CustomShardRangeLoader | null;
  customStreamLoader?: CustomShardStreamLoader | null;
  verifyHashes?: boolean;
  manifest?: RDRRManifest | null;
  loadingConfig?: ShardCacheConfigSchema;
  maxConcurrentLoads?: number;
}

export class ShardCache {
  lastSource: ShardSourceInfo | null;

  constructor(config: ShardCacheConfig);
  configure(config: Partial<ShardCacheConfig>): void;
  setCustomLoader(
    loader: CustomShardLoader | null,
    verify?: boolean,
    options?: {
      loadRange?: CustomShardRangeLoader | null;
      streamRange?: CustomShardStreamLoader | null;
    }
  ): void;
  setManifest(manifest: RDRRManifest | null): void;
  get hasCustomLoader(): boolean;
  get hasCustomRangeLoader(): boolean;
  get hasCustomStreamLoader(): boolean;
  get canStreamRanges(): boolean;
  has(shardIndex: number): boolean;
  get size(): number;
  get totalBytes(): number;
  load(shardIndex: number, options?: ShardLoadOptions): Promise<ArrayBuffer>;
  loadRange(shardIndex: number, offset?: number, length?: number | null, options?: ShardLoadOptions): Promise<ArrayBuffer>;
  streamRange(
    shardIndex: number,
    offset?: number,
    length?: number | null,
    options?: CustomShardStreamOptions
  ): AsyncIterable<Uint8Array>;
  prefetch(shardIndex: number): Promise<ArrayBuffer>;
  clear(): void;
  configureForModel(manifest: RDRRManifest | null, hasCustomLoader: boolean): void;
}

export function createShardCache(
  maxEntries?: number,
  loadingConfig?: ShardCacheConfigSchema
): ShardCache;
