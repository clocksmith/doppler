import type { DopplerCapsule } from '../../config/capsule.js';
import type { CapsuleV2Artifact } from '../../config/capsule-v2.js';
export declare function createCapsuleArtifactSource(capsule: DopplerCapsule, artifactStore: {
  readArtifact(artifact: CapsuleV2Artifact): Promise<Uint8Array>;
  readArtifactRange?(artifact: CapsuleV2Artifact, offset: number, length: number): Promise<Uint8Array>;
}): Promise<{
  modelId: string; manifest: Record<string, unknown>; manifestText: string; manifestHash: string; storageContext: object;
}>;
