export interface DopplerRevocationTargets {
  logicalModelIds: readonly string[];
  modelIds: readonly string[];
  sourceCheckpointIds: readonly string[];
  weightPackIds: readonly string[];
  manifestVariantIds: readonly string[];
  artifactVariantIds: readonly `sha256:${string}`[];
  adapterIds: readonly string[];
  adapterDigests: readonly `sha256:${string}`[];
}

export interface DopplerRevocationRecord {
  id: string;
  state: 'revoked';
  issuedAtUtc: string;
  severity: 'correctness' | 'security' | 'reliability' | 'provenance' | 'policy';
  reason: string;
  targets: DopplerRevocationTargets;
  replacements: DopplerRevocationTargets;
  evidencePaths: readonly string[];
}

export interface DopplerRevocationRegistry {
  $schema: 'schema/revocation-registry.schema.json';
  schemaVersion: 1;
  source: 'doppler';
  updatedAtUtc: string;
  trust: {
    distribution: 'bundled-package';
    signatureVerification: 'unavailable';
  };
  revocations: readonly DopplerRevocationRecord[];
}

export interface DopplerRevocationIdentity {
  logicalModelId?: string | null;
  modelId?: string | null;
  sourceCheckpointId?: string | null;
  weightPackId?: string | null;
  manifestVariantId?: string | null;
  artifactVariantId?: string | null;
  adapterId?: string | null;
  adapterDigest?: string | null;
}

export declare function validateRevocationRegistry(value: unknown): DopplerRevocationRegistry;
export declare function loadRevocationRegistry(): Promise<DopplerRevocationRegistry>;
export declare function findResolutionRevocation(
  identity: DopplerRevocationIdentity,
  registry: DopplerRevocationRegistry
): {
  revocation: DopplerRevocationRecord;
  matchedFields: readonly string[];
} | null;
export declare function assertResolutionNotRevoked(
  identity: DopplerRevocationIdentity,
  registry: DopplerRevocationRegistry
): void;
export declare function assertBundledResolutionNotRevoked(identity: DopplerRevocationIdentity): Promise<void>;
export declare function authorizeBundledAdapter<T extends object>(adapter: T): Promise<T>;
export declare function assertBundledAdapterAuthorized(adapter: object | null | undefined): void;
