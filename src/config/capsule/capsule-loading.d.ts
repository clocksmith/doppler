export interface CapsuleLoadingPolicy {
  loadTimeoutMs: number | null; maxMetadataBytes: number; maxRetainedArtifactBytes: number | null;
  maxVerifiedBackingBytes: number | null; artifactHashBackend: 'host' | 'javascript' | 'node-crypto';
  maxAcquisitionChunkBytes: number; verificationYieldBytes: number;
}
export declare function normalizeCapsuleLoadingPolicy(options: Partial<CapsuleLoadingPolicy>): Readonly<CapsuleLoadingPolicy>;
