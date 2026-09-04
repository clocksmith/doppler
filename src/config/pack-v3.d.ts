import type { DopplerPackV2 } from './pack-v2.js';
import type { PackSigner } from './pack-signature.js';
export const PACK_V3_SCHEMA_ID: 'doppler.pack/v3';
export const PACK_V3_SCHEMA_VERSION: 3;
export interface DopplerPackV3 extends Omit<DopplerPackV2, 'schema' | 'schemaVersion' | 'createdAtUtc' | 'release'> {
  schema: 'doppler.pack/v3';
  schemaVersion: 3;
}
export declare function getPackV3SemanticPayload(pack: DopplerPackV3 | DopplerPackV2): Record<string, unknown>;
export declare function hashPackV3(pack: DopplerPackV3 | DopplerPackV2): `sha256:${string}`;
export declare function validatePackV3(pack: unknown, options?: { requireSignature?: boolean }): { ok: boolean; errors: string[] };
export declare function buildPackV3(executable: Pick<DopplerPackV2, 'modelId' | 'modelIR' | 'targetPlans' | 'wgslModules' | 'artifacts' | 'program'>): DopplerPackV3;
export declare function signPackV3(pack: DopplerPackV3, signer: PackSigner): Promise<DopplerPackV3>;
export declare function verifyPackV3Signature(pack: DopplerPackV3, trustedSigners: Map<string, JsonWebKey> | Record<string, JsonWebKey>): Promise<true>;
export declare function migratePackV2(pack: DopplerPackV2, options: { trustedSigners: Map<string, JsonWebKey> | Record<string, JsonWebKey>; signer: PackSigner }): Promise<{
  pack: DopplerPackV3;
  release: DopplerPackV2['release'];
  migratedFrom: { schema: 'doppler.pack/v2'; semanticRoot: string; envelopeDigest: string };
}>;
