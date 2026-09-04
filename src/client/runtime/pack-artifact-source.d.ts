import type { DopplerPack } from '../../config/pack.js';
import type { PackV2Artifact } from '../../config/pack-v2.js';
export declare function createPackArtifactSource(pack: DopplerPack, artifactStore: { readArtifact(artifact: PackV2Artifact): Promise<Uint8Array> }): Promise<{
  modelId: string; manifest: Record<string, unknown>; manifestText: string; manifestHash: string; storageContext: object;
}>;
