import type { ModelIR } from './model-ir.js';
import type { PackReleaseContract } from './pack-release-contract.js';
import type { TargetPlan } from './target-plan.js';

export const PACK_V2_SCHEMA_ID: 'doppler.pack/v2';
export const PACK_V2_SCHEMA_VERSION: 2;
export const PACK_V2_PROGRAM_SCHEMA_ID: 'doppler.pack-program/v1';
export const PACK_V2_SIGNATURE_ALGORITHM: 'Ed25519';

export interface PackV2Artifact {
  artifactId: string;
  role: string;
  path: string;
  hash: `sha256:${string}`;
  sizeBytes: number;
}

export interface PackV2WgslModule {
  id: string;
  file: string;
  entry: string;
  digest: `sha256:${string}`;
  sourceHash: `sha256:${string}`;
  sourceArtifactId: string;
  metadata?: Record<string, unknown>;
}

export interface DopplerPackV2 {
  schema: 'doppler.pack/v2';
  schemaVersion: 2;
  packId: string;
  modelId: string;
  createdAtUtc: string;
  semanticRoot: `sha256:${string}`;
  modelIR: ModelIR;
  targetPlans: TargetPlan[];
  wgslModules: PackV2WgslModule[];
  artifacts: PackV2Artifact[];
  program: Record<string, unknown> & {
    schema: 'doppler.pack-program/v1';
    programBundleHash: `sha256:${string}`;
    programBundleArtifactId: string;
    executionGraphHash: `sha256:${string}`;
    manifestArtifactId: string;
    modelIREvidenceArtifactId?: string;
    tokenizerArtifactIds: string[];
    weightArtifactIds: string[];
  };
  release: PackReleaseContract;
  signature: null | {
    authority: string;
    algorithm: 'Ed25519';
    publicKeyDigest: `sha256:${string}`;
    signatureHex: string;
    signedDigest: `sha256:${string}`;
  };
}

export declare function getPackV2SemanticPayload(pack: DopplerPackV2): Record<string, unknown>;
export declare function hashPackV2(pack: DopplerPackV2): `sha256:${string}`;
export declare function hashPackV2Envelope(pack: DopplerPackV2): `sha256:${string}`;
export declare function hashPackV2PublicKey(publicKeyJwk: JsonWebKey): `sha256:${string}`;
export declare function validatePackV2(pack: unknown, options?: { requireSignature?: boolean }): { ok: boolean; errors: string[] };
export declare function validatePackExecutable(pack: unknown): { ok: boolean; errors: string[] };
export declare function buildPackV2(params: Omit<DopplerPackV2, 'schema' | 'schemaVersion' | 'packId' | 'semanticRoot' | 'signature'>): DopplerPackV2;
export declare function signPackV2(pack: DopplerPackV2, signer: { authority: string; privateKeyJwk: JsonWebKey; publicKeyJwk: JsonWebKey }): Promise<DopplerPackV2>;
export declare function verifyPackV2Signature(pack: DopplerPackV2, trustedSigners: Map<string, JsonWebKey> | Record<string, JsonWebKey>): Promise<true>;
export declare function verifyPackV2Artifacts(pack: DopplerPackV2, artifactStore: { hashArtifact(artifact: PackV2Artifact): Promise<{ hash: string; sizeBytes: number }> }): Promise<Array<Record<string, unknown>>>;
export declare function verifyPackV2(pack: DopplerPackV2, options: { trustedSigners: Map<string, JsonWebKey> | Record<string, JsonWebKey>; artifactStore: object }): Promise<{ pack: DopplerPackV2; artifactReceipts: Array<Record<string, unknown>> }>;
export declare function freezePackV2<T>(value: T): T;
