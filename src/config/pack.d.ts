import type { DopplerPackV2 } from './pack-v2.js';
import type { PackV2Artifact } from './pack-v2.js';
import type { DopplerPackV3 } from './pack-v3.js';
import type { PackReleaseEvent, PackReleasePolicy, verifyPackReleaseEvents } from './pack-release-events.js';
export type DopplerPack = DopplerPackV2 | DopplerPackV3;
export interface PackIdentity { schema: DopplerPack['schema']; packId: string; semanticRoot: string; envelopeDigest: string; artifactClosureDigest: string }
export declare function validatePack(pack: unknown, options?: { requireSignature?: boolean }): { ok: boolean; errors: string[] };
export declare function getPackIdentity(pack: DopplerPack): PackIdentity;
export declare function verifyPack(pack: DopplerPack, options: {
  trustedSigners: Map<string, JsonWebKey> | Record<string, JsonWebKey>;
  artifactStore: { readArtifact(artifact: PackV2Artifact): Promise<Uint8Array | ArrayBuffer> };
  releaseEvents?: PackReleaseEvent[];
  releaseTrustedSigners?: Map<string, JsonWebKey> | Record<string, JsonWebKey>;
  releasePolicy?: PackReleasePolicy;
}): Promise<{ pack: DopplerPack; identity: PackIdentity; artifactReceipts: Array<Record<string, unknown>>; lifecycle: Awaited<ReturnType<typeof verifyPackReleaseEvents>> | null }>;
