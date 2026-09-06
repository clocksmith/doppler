import type { DopplerCapsuleV2 } from './capsule-v2.js';
import type { CapsuleSigner } from './capsule-signature.js';
export const CAPSULE_V3_SCHEMA_ID: 'doppler.capsule/v3';
export const CAPSULE_V3_SCHEMA_VERSION: 3;
export interface DopplerCapsuleV3 extends Omit<DopplerCapsuleV2, 'schema' | 'schemaVersion' | 'createdAtUtc' | 'release'> {
  schema: 'doppler.capsule/v3';
  schemaVersion: 3;
}
export declare function getCapsuleV3SemanticPayload(capsule: DopplerCapsuleV3 | DopplerCapsuleV2): Record<string, unknown>;
export declare function hashCapsuleV3(capsule: DopplerCapsuleV3 | DopplerCapsuleV2): `sha256:${string}`;
export declare function validateCapsuleV3(capsule: unknown, options?: { requireSignature?: boolean }): { ok: boolean; errors: string[] };
export declare function buildCapsuleV3(executable: Pick<DopplerCapsuleV2, 'modelId' | 'modelIR' | 'targetPlans' | 'wgslModules' | 'artifacts' | 'program'>): DopplerCapsuleV3;
export declare function signCapsuleV3(capsule: DopplerCapsuleV3, signer: CapsuleSigner): Promise<DopplerCapsuleV3>;
export declare function verifyCapsuleV3Signature(capsule: DopplerCapsuleV3, trustedSigners: Map<string, JsonWebKey> | Record<string, JsonWebKey>): Promise<true>;
export declare function migrateCapsuleV2(capsule: DopplerCapsuleV2, options: { trustedSigners: Map<string, JsonWebKey> | Record<string, JsonWebKey>; signer: CapsuleSigner }): Promise<{
  capsule: DopplerCapsuleV3;
  release: DopplerCapsuleV2['release'];
  migratedFrom: { schema: 'doppler.capsule/v2'; semanticRoot: string; envelopeDigest: string };
}>;
