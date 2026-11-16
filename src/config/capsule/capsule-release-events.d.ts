import type { DopplerCapsuleV2 } from './capsule-v2.js';
import type { DopplerCapsuleV3 } from './capsule-v3.js';
import type { CapsuleSigner, CapsuleSignature } from './capsule-signature.js';
export const CAPSULE_RELEASE_EVENT_SCHEMA: 'doppler.capsule-release-event/v1';
export interface CapsuleReference { schema: 'doppler.capsule/v2' | 'doppler.capsule/v3'; semanticRoot: string; envelopeDigest: string }
export interface ReleaseCheckpoint { sequence: number; digest: string | null }
export interface CapsuleRetainedLocalUse {
  schema: 'doppler.capsule-retained-local-use/v1';
  capsule: CapsuleReference;
  releaseEventDigest: string;
  applicationDigest: string;
  acceptedAtUtc: string;
  acknowledgeUnseenRevocations: true;
}
export interface CapsuleReleasePolicy {
  now: string;
  minimumSequence: number;
  checkpoint: ReleaseCheckpoint;
  retainedLocalUse?: CapsuleRetainedLocalUse;
}
export interface CapsuleReleaseAuthorization {
  mode: 'managed' | 'retained-local';
  verifiedAtUtc: string;
  eventExpired: boolean;
  unseenRevocations: 'unknown';
  retainedLocalUse: CapsuleRetainedLocalUse | null;
}
export declare class CapsuleReleaseStateError extends Error {
  readonly checkpoint: ReleaseCheckpoint;
  constructor(cause: Error, checkpoint: ReleaseCheckpoint);
}
export interface CapsuleReleaseEvent {
  schema: 'doppler.capsule-release-event/v1';
  capsule: CapsuleReference;
  sequence: number;
  previousEventDigest: string | null;
  issuedAtUtc: string;
  expiresAtUtc: string;
  action: 'eligible' | 'blocked' | 'promoted' | 'quarantined' | 'revoked' | 'superseded' | 'rollback-authorized';
  release: DopplerCapsuleV2['release'];
  migratedFrom: CapsuleReference | null;
  nextSigner: JsonWebKey | null;
  digest: string;
  signature: CapsuleSignature | null;
}
export declare function hashCapsuleReleaseEvent(event: CapsuleReleaseEvent): `sha256:${string}`;
export declare function validateCapsuleReleaseEvent(event: unknown, options?: { requireSignature?: boolean }): { ok: boolean; errors: string[] };
export declare function signCapsuleReleaseEvent(params: Omit<CapsuleReleaseEvent, 'schema' | 'digest' | 'signature'>, signer: CapsuleSigner): Promise<CapsuleReleaseEvent>;
export declare function verifyCapsuleReleaseEvents(events: CapsuleReleaseEvent[], options: {
  capsule: DopplerCapsuleV2 | DopplerCapsuleV3;
  trustedSigners: Map<string, JsonWebKey> | Record<string, JsonWebKey>;
  policy: CapsuleReleasePolicy;
}): Promise<{ release: DopplerCapsuleV2['release']; event: CapsuleReleaseEvent; checkpoint: ReleaseCheckpoint; authorization: CapsuleReleaseAuthorization; nextPublicKeyDigest: string }>;
