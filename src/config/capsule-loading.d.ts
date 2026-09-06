export interface CapsuleLoadingPolicy { loadTimeoutMs: number | null; maxMetadataBytes: number }
export declare function normalizeCapsuleLoadingPolicy(options: Partial<CapsuleLoadingPolicy>): Readonly<CapsuleLoadingPolicy>;
