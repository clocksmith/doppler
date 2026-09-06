import type { ModelIR } from './model-ir.js';
import type { CapsuleReleaseContract } from './capsule-release-contract.js';
import type { TargetPlan } from './target-plan.js';

export const CAPSULE_V2_SCHEMA_ID: 'doppler.capsule/v2';
export const CAPSULE_V2_SCHEMA_VERSION: 2;
export const CAPSULE_V2_PROGRAM_SCHEMA_ID: 'doppler.capsule-program/v1';
export const CAPSULE_V2_SIGNATURE_ALGORITHM: 'Ed25519';

export interface CapsuleV2Artifact {
  artifactId: string;
  role: string;
  path: string;
  hash: `sha256:${string}`;
  sizeBytes: number;
}

export interface CapsuleV2WgslModule {
  id: string;
  file: string;
  entry: string;
  digest: `sha256:${string}`;
  sourceHash: `sha256:${string}`;
  sourceArtifactId: string;
  metadata?: Record<string, unknown>;
}

export interface DopplerCapsuleV2 {
  schema: 'doppler.capsule/v2';
  schemaVersion: 2;
  capsuleId: string;
  modelId: string;
  createdAtUtc: string;
  semanticRoot: `sha256:${string}`;
  modelIR: ModelIR;
  targetPlans: TargetPlan[];
  wgslModules: CapsuleV2WgslModule[];
  artifacts: CapsuleV2Artifact[];
  program: Record<string, unknown> & {
    schema: 'doppler.capsule-program/v1';
    programBundleHash: `sha256:${string}`;
    programBundleArtifactId: string;
    executionGraphHash: `sha256:${string}`;
    manifestArtifactId: string;
    modelIREvidenceArtifactId?: string;
    tokenizerArtifactIds: string[];
    weightArtifactIds: string[];
  };
  release: CapsuleReleaseContract;
  signature: null | {
    authority: string;
    algorithm: 'Ed25519';
    publicKeyDigest: `sha256:${string}`;
    signatureHex: string;
    signedDigest: `sha256:${string}`;
  };
}

export declare function getCapsuleV2SemanticPayload(capsule: DopplerCapsuleV2): Record<string, unknown>;
export declare function hashCapsuleV2(capsule: DopplerCapsuleV2): `sha256:${string}`;
export declare function hashCapsuleV2Envelope(capsule: DopplerCapsuleV2): `sha256:${string}`;
export declare function hashCapsuleV2PublicKey(publicKeyJwk: JsonWebKey): `sha256:${string}`;
export declare function validateCapsuleV2(capsule: unknown, options?: { requireSignature?: boolean }): { ok: boolean; errors: string[] };
export declare function validateCapsuleExecutable(capsule: unknown): { ok: boolean; errors: string[] };
export declare function buildCapsuleV2(params: Omit<DopplerCapsuleV2, 'schema' | 'schemaVersion' | 'capsuleId' | 'semanticRoot' | 'signature'>): DopplerCapsuleV2;
export declare function signCapsuleV2(capsule: DopplerCapsuleV2, signer: { authority: string; privateKeyJwk: JsonWebKey; publicKeyJwk: JsonWebKey }): Promise<DopplerCapsuleV2>;
export declare function verifyCapsuleV2Signature(capsule: DopplerCapsuleV2, trustedSigners: Map<string, JsonWebKey> | Record<string, JsonWebKey>): Promise<true>;
export declare function verifyCapsuleV2Artifacts(capsule: DopplerCapsuleV2, artifactStore: { hashArtifact(artifact: CapsuleV2Artifact): Promise<{ hash: string; sizeBytes: number }> }): Promise<Array<Record<string, unknown>>>;
export declare function verifyCapsuleV2(capsule: DopplerCapsuleV2, options: { trustedSigners: Map<string, JsonWebKey> | Record<string, JsonWebKey>; artifactStore: object }): Promise<{ capsule: DopplerCapsuleV2; artifactReceipts: Array<Record<string, unknown>> }>;
export declare function freezeCapsuleV2<T>(value: T): T;
