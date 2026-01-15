import type { RDRRManifest } from '../storage/rdrr-format.js';
import type { ShardCacheConfigSchema } from '../config/schema/loading.schema.js';
import type { CustomShardLoader, ShardSourceInfo } from './loader-types.js';

export interface ShardCacheConfig {
  maxEntries: number;
  customLoader?: CustomShardLoader | null;
  verifyHashes?: boolean;
  manifest?: RDRRManifest | null;
  loadingConfig?: ShardCacheConfigSchema;
}

export class ShardCache {
  lastSource: ShardSourceInfo | null;

  constructor(config: ShardCacheConfig);
  configure(config: Partial<ShardCacheConfig>): void;
  setCustomLoader(loader: CustomShardLoader | null, verify?: boolean): void;
  setManifest(manifest: RDRRManifest | null): void;
  get hasCustomLoader(): boolean;
  has(shardIndex: number): boolean;
  get size(): number;
  get totalBytes(): number;
  load(shardIndex: number): Promise<ArrayBuffer>;
  clear(): void;
  configureForModel(manifest: RDRRManifest | null, hasCustomLoader: boolean): void;
}

export function createShardCache(
  maxEntries?: number,
  loadingConfig?: ShardCacheConfigSchema
): ShardCache;
