export declare const DEFAULT_HF_REGISTRY_PATH: string;
export declare const DEFAULT_HF_REGISTRY_URL: string;
export declare function buildHfResolveUrl(repoId: unknown, revision: unknown, repoPath: unknown): string;
export declare function getEntryHfSpec(entry: unknown): {
  repoId: string;
  revision: string;
  path: string;
  complete: boolean;
};
export declare function buildEntryRemoteBaseUrl(entry: unknown): string | null;
export declare function resolveDemoRegistryEntryBaseUrl(entry: unknown, catalogSourceUrl: unknown): string | null;
export declare function shouldDemoSurfaceRemoteRegistryEntry(entry: unknown, catalogSourceUrl: unknown): boolean;
export declare function buildManifestUrl(baseUrl: unknown): string;
export declare function buildShardUrl(baseUrl: unknown, shard: unknown): string;
export declare function extractCommitShaFromUrl(value: unknown): string | null;
