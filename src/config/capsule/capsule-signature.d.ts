import type { DopplerCapsuleV2 } from './capsule-v2.js';
export type CapsuleSignature = NonNullable<DopplerCapsuleV2['signature']>;
export interface CapsuleSigner { authority: string; publicKeyJwk: JsonWebKey; privateKeyJwk: JsonWebKey }
export declare function validateCapsuleSignature(signature: unknown, digest: string): string[];
export declare function signCapsuleDigest(digest: string, signer: CapsuleSigner): Promise<CapsuleSignature>;
export declare function verifyCapsuleDigest(signature: CapsuleSignature, digest: string, publicKeyJwk: JsonWebKey): Promise<true>;
