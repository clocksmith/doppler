import type { DopplerPackV2 } from './pack-v2.js';
import type { DopplerPackV3 } from './pack-v3.js';
import type { PackSigner, PackSignature } from './pack-signature.js';
export const PACK_RELEASE_EVENT_SCHEMA: 'doppler.pack-release-event/v1';
export interface PackReference { schema: 'doppler.pack/v2' | 'doppler.pack/v3'; semanticRoot: string; envelopeDigest: string }
export interface ReleaseCheckpoint { sequence: number; digest: string | null }
export interface PackReleasePolicy { now: string; minimumSequence: number; checkpoint: ReleaseCheckpoint }
export interface PackReleaseEvent {
  schema: 'doppler.pack-release-event/v1';
  pack: PackReference;
  sequence: number;
  previousEventDigest: string | null;
  issuedAtUtc: string;
  expiresAtUtc: string;
  action: 'eligible' | 'blocked' | 'promoted' | 'quarantined' | 'revoked' | 'superseded' | 'rollback-authorized';
  release: DopplerPackV2['release'];
  migratedFrom: PackReference | null;
  nextSigner: JsonWebKey | null;
  digest: string;
  signature: PackSignature | null;
}
export declare function hashPackReleaseEvent(event: PackReleaseEvent): `sha256:${string}`;
export declare function validatePackReleaseEvent(event: unknown, options?: { requireSignature?: boolean }): { ok: boolean; errors: string[] };
export declare function signPackReleaseEvent(params: Omit<PackReleaseEvent, 'schema' | 'digest' | 'signature'>, signer: PackSigner): Promise<PackReleaseEvent>;
export declare function verifyPackReleaseEvents(events: PackReleaseEvent[], options: {
  pack: DopplerPackV2 | DopplerPackV3;
  trustedSigners: Map<string, JsonWebKey> | Record<string, JsonWebKey>;
  policy: PackReleasePolicy;
}): Promise<{ release: DopplerPackV2['release']; event: PackReleaseEvent; checkpoint: ReleaseCheckpoint; nextPublicKeyDigest: string }>;
