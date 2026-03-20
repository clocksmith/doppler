export declare const DEFAULT_HF_REPO_ID: string;
export declare const DEFAULT_HF_REGISTRY_PATH: string;
export declare const DEFAULT_HF_REGISTRY_URL: string;
export declare const DEFAULT_EXTERNAL_MODELS_ROOT: string;
export declare const DEFAULT_EXTERNAL_VOLUME_INDEX_PATH: string;

export declare function normalizeText(value: unknown): string;
export declare function normalizeToken(value: unknown): string;
export declare function normalizeRepoPath(value: unknown): string;
export declare function detectDefaultExternalModelsRoot(): string;
export declare function isPlainObject(value: unknown): boolean;
export declare function ensureCatalogPayload(payload: unknown, label?: string): {
  models: unknown[];
  [key: string]: unknown;
};
export declare function loadJsonFile(
  filePath: string,
  label?: string
): Promise<{
  models: unknown[];
  [key: string]: unknown;
}>;
export declare function writeJsonFile(filePath: string, payload: unknown): Promise<void>;
export declare function buildHfResolveUrl(repoId: unknown, revision: unknown, repoPath: unknown): string;
export declare function getEntryHfSpec(entry: unknown): {
  repoId: string;
  revision: string;
  path: string;
};
export declare function buildEntryRemoteBaseUrl(entry: unknown): string | null;
export declare function resolveDemoRegistryEntryBaseUrl(entry: unknown, catalogSourceUrl: unknown): string | null;
export declare function shouldDemoSurfaceRemoteRegistryEntry(entry: unknown, catalogSourceUrl: unknown): boolean;
export declare function buildManifestUrl(baseUrl: unknown): string;
export declare function buildShardUrl(baseUrl: unknown, shard: unknown): string;
export declare function collectDuplicateModelIds(models: unknown[]): string[];
export declare function sortCatalogEntries(models: unknown[]): unknown[];
export declare function findCatalogEntry(
  payload: { models?: unknown[] } | null | undefined,
  modelId: unknown
): unknown | null;
export declare function isHostedRegistryApprovedEntry(entry: unknown): boolean;
export declare function buildPublishedRegistryEntry(localEntry: unknown, revision: unknown): Record<string, unknown>;
export declare function buildHostedRegistryPayload(
  payload: unknown,
  revisionOverrides?: Map<string, string>
): Record<string, unknown>;
export declare function extractCommitShaFromUrl(value: unknown): string | null;
export declare function validateLocalHfEntryShape(entry: unknown): void;
export declare function probeUrl(
  url: string,
  options?: Record<string, unknown>
): Promise<Record<string, unknown>>;
export declare function fetchJson(
  url: string,
  options?: Record<string, unknown>
): Promise<Record<string, unknown>>;
export declare function fetchRepoHeadSha(
  repoId: string,
  options?: Record<string, unknown>
): Promise<string | null>;
