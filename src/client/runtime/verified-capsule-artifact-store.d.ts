import type { DopplerCapsule } from '../../config/capsule.js';
import type { CapsuleV2Artifact } from '../../config/capsule-v2.js';
import type { CapsuleAcquisitionOptions, CapsuleArtifactReader } from './capsule-acquisition.js';
declare const backingBrand: unique symbol;
export interface CapsuleArtifactBacking { readonly [backingBrand]: true }
/** Opaque host owner. Verified blocks live only while stores hold leases. */
export declare function createCapsuleArtifactBacking(): CapsuleArtifactBacking;
/** Only factory-created, immutable store interfaces authenticate owned snapshot receipts. */
export declare function isVerifiedCapsuleArtifactStore(store: unknown): store is ReturnType<typeof createVerifiedCapsuleArtifactStore>;
export declare function createVerifiedCapsuleArtifactStore(capsule: DopplerCapsule, source: CapsuleArtifactReader, options?: CapsuleAcquisitionOptions, backing?: CapsuleArtifactBacking): Readonly<{
  readArtifact(artifact: CapsuleV2Artifact): Promise<Uint8Array>;
  readArtifactRange(artifact: CapsuleV2Artifact, offset: number, length: number): Promise<Uint8Array>;
  hashArtifact(artifact: CapsuleV2Artifact): Promise<{ hash: string; sizeBytes: number }>;
  getMetrics(): Readonly<{ sourceBytes: number; hashedBytes: number; copiedBytes: number; retainedBytes: number; peakRetainedBytes: number; returnedBytes: number;
    evictions: number; sourceReadMs: number; hashingMs: number; copyingMs: number;
    backingBytes: number; peakBackingBytes: number; backingFiles: number; snapshotCopiedBytes: number;
    sharedBackingBytes: number; peakSnapshotBlockBytes: number;
    peakSourceChunkBytes: number; streamedSourceBytes: number; verificationYields: number }>;
  close(): void;
}>;
