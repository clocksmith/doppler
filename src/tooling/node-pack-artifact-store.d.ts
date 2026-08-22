import type { PackV2Artifact } from './pack-v2.js';

export interface NodePackArtifactStore {
  hashArtifact(artifact: PackV2Artifact): Promise<{ hash: string; sizeBytes: number }>;
  readArtifact(artifact: PackV2Artifact): Promise<Uint8Array>;
  resolveArtifactPath(artifact: PackV2Artifact): string;
  resolveArtifactUrl(artifact: PackV2Artifact): string;
}

export declare function createNodePackArtifactStore(packPath: string): NodePackArtifactStore;
