import type { DopplerPackV2 } from './pack-v2.js';
export type PackSignature = NonNullable<DopplerPackV2['signature']>;
export interface PackSigner { authority: string; publicKeyJwk: JsonWebKey; privateKeyJwk: JsonWebKey }
export declare function validatePackSignature(signature: unknown, digest: string): string[];
export declare function signPackDigest(digest: string, signer: PackSigner): Promise<PackSignature>;
export declare function verifyPackDigest(signature: PackSignature, digest: string, publicKeyJwk: JsonWebKey): Promise<true>;
