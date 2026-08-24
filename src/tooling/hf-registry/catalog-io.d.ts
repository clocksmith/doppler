export declare const DEFAULT_EXTERNAL_MODELS_ROOT: string;
export declare function normalizeText(value: unknown): string;
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
export declare function collectDuplicateModelIds(models: unknown[]): string[];
export declare function findCatalogEntry(
  payload: { models?: unknown[] } | null | undefined,
  modelId: unknown
): unknown | null;
