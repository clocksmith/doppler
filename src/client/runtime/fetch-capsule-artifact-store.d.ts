import type { CapsuleV2Artifact } from '../../tooling/capsule-v2.js';
import type { CapsuleAcquisitionOptions } from './capsule-acquisition.js';

export interface FetchCapsuleArtifactStore {
  hashArtifact(artifact: CapsuleV2Artifact, options?: CapsuleAcquisitionOptions): Promise<{ hash: string; sizeBytes: number }>;
  readArtifact(artifact: CapsuleV2Artifact, options?: CapsuleAcquisitionOptions): Promise<Uint8Array>;
  resolveArtifactUrl(artifact: CapsuleV2Artifact): string;
}

export declare function createFetchCapsuleArtifactStore(capsuleUrl: string): FetchCapsuleArtifactStore;
