import type { DopplerCapsule } from '../../config/capsule.js';
import type { CapsuleV2Artifact } from '../../config/capsule-v2.js';
import type { CapsuleAcquisitionOptions } from './capsule-acquisition.js';
export declare function createVerifiedCapsuleArtifactStore(capsule: DopplerCapsule, source: { readArtifact(artifact: CapsuleV2Artifact, options?: CapsuleAcquisitionOptions): Promise<Uint8Array | ArrayBuffer> }, options?: CapsuleAcquisitionOptions): {
  readArtifact(artifact: CapsuleV2Artifact): Promise<Uint8Array>;
  readArtifactRange(artifact: CapsuleV2Artifact, offset: number, length: number): Promise<Uint8Array>;
  hashArtifact(artifact: CapsuleV2Artifact): Promise<{ hash: string; sizeBytes: number }>;
  getMetrics(): Readonly<{ sourceBytes: number; hashedBytes: number; copiedBytes: number; retainedBytes: number; peakRetainedBytes: number; returnedBytes: number;
    evictions: number; sourceReadMs: number; hashingMs: number; copyingMs: number }>;
  close(): void;
};
