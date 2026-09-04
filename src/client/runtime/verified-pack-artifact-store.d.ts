import type { DopplerPack } from '../../config/pack.js';
import type { PackV2Artifact } from '../../config/pack-v2.js';
export declare function createVerifiedPackArtifactStore(pack: DopplerPack, source: { readArtifact(artifact: PackV2Artifact): Promise<Uint8Array | ArrayBuffer> }): {
  readArtifact(artifact: PackV2Artifact): Promise<Uint8Array>;
  hashArtifact(artifact: PackV2Artifact): Promise<{ hash: string; sizeBytes: number }>;
  close(): void;
};
