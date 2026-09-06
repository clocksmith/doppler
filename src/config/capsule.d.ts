import type { DopplerCapsuleV2 } from './capsule-v2.js';
import type { CapsuleV2Artifact } from './capsule-v2.js';
import type { DopplerCapsuleV3 } from './capsule-v3.js';
import type { CapsuleReleaseEvent, CapsuleReleasePolicy, verifyCapsuleReleaseEvents } from './capsule-release-events.js';
export type DopplerCapsule = DopplerCapsuleV2 | DopplerCapsuleV3;
export interface CapsuleIdentity { schema: DopplerCapsule['schema']; capsuleId: string; semanticRoot: string; envelopeDigest: string; artifactClosureDigest: string }
export declare function validateCapsule(capsule: unknown, options?: { requireSignature?: boolean }): { ok: boolean; errors: string[] };
export declare function getCapsuleIdentity(capsule: DopplerCapsule): CapsuleIdentity;
export interface CapsuleMetadataVerificationOptions {
  trustedSigners: Map<string, JsonWebKey> | Record<string, JsonWebKey>;
  releaseEvents?: CapsuleReleaseEvent[];
  releaseTrustedSigners?: Map<string, JsonWebKey> | Record<string, JsonWebKey>;
  releasePolicy?: CapsuleReleasePolicy;
}
export interface VerifiedCapsuleMetadata {
  capsule: DopplerCapsule; identity: CapsuleIdentity; lifecycle: Awaited<ReturnType<typeof verifyCapsuleReleaseEvents>> | null;
}
/** Authenticates metadata and release policy only. Does not establish artifact integrity or authorize execution. */
export declare function verifyCapsuleMetadata(capsule: DopplerCapsule, options: CapsuleMetadataVerificationOptions): Promise<VerifiedCapsuleMetadata>;
export declare function verifyCapsule(capsule: DopplerCapsule, options: CapsuleMetadataVerificationOptions & {
  artifactStore: { readArtifact(artifact: CapsuleV2Artifact): Promise<Uint8Array | ArrayBuffer> };
}): Promise<VerifiedCapsuleMetadata & { artifactReceipts: Array<Record<string, unknown>> }>;
