export interface CapsuleLoadingPolicy { loadTimeoutMs: number | null; maxMetadataBytes: number; maxRetainedArtifactBytes: number | null }
export declare function normalizeCapsuleLoadingPolicy(options: Partial<CapsuleLoadingPolicy>): Readonly<CapsuleLoadingPolicy>;
