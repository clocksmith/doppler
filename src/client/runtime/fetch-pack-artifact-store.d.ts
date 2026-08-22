import type { PackV2Artifact } from '../../tooling/pack-v2.js';

export interface FetchPackArtifactStore {
  hashArtifact(artifact: PackV2Artifact): Promise<{ hash: string; sizeBytes: number }>;
  readArtifact(artifact: PackV2Artifact): Promise<Uint8Array>;
  resolveArtifactUrl(artifact: PackV2Artifact): string;
}

export declare function createFetchPackArtifactStore(packUrl: string): FetchPackArtifactStore;
