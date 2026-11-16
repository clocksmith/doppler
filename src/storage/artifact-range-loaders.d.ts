import type { CreateArtifactStorageContextOptions, ArtifactStorageContext } from './artifact-storage-context.js';
export function createArtifactRangeLoaders(options: {
  shardSources: Array<{ path: string; size: number }>;
  readRange: CreateArtifactStorageContextOptions['readRange'];
  readInto: CreateArtifactStorageContextOptions['readInto'] | null;
  ensureVerifiedSource: (path: string) => Promise<void>;
}): Pick<ArtifactStorageContext, 'loadShardRange' | 'loadShardRangeInto'>;
