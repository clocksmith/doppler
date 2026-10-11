import type { DopplerCapsule } from '../../config/capsule.js';
import type { CapsuleV2Artifact } from '../../config/capsule-v2.js';
import type { VerifiedPieceReceipt } from '../../storage/verified-piece-reader.js';
import type { CapsuleArtifactReader, CapsuleAcquisitionOptions } from './capsule-acquisition.js';
export function createCapsulePieceReader(capsule: DopplerCapsule,
  verifiedStore: { readArtifact(artifact: CapsuleV2Artifact): Promise<Uint8Array> },
  source: CapsuleArtifactReader, options: CapsuleAcquisitionOptions): Promise<Readonly<{
    hasArtifact(artifact: CapsuleV2Artifact): boolean;
    readRange(artifact: CapsuleV2Artifact, offset: number, length: number): Promise<Uint8Array>;
    getReceipt(): VerifiedPieceReceipt & { mode: 'signed-piece-index'; indexArtifactId: string; declaredWeightArtifactIds: string[] };
    close(): Promise<void>;
  }>>;
