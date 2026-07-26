export interface DemoCatalogEntry {
  modelId: string;
  label?: string;
  source?: string;
  urls?: string[];
  files?: Record<string, string>;
  weightsRef?: {
    weightPackId?: string;
    primaryModelId?: string;
  } | null;
  [key: string]: unknown;
}

export declare function canRemoveModelStatus(status: string | null | undefined): boolean;

export declare function buildModelCardDetail(
  entry: DemoCatalogEntry,
  status: string | null | undefined
): string;

export declare function setModelCallbacks(callbacks: {
  onLoaded?: ((model: unknown, modelId: string) => void) | null;
  onDownloadProgress?: ((progress: unknown) => void) | null;
}): void;

export declare function selectDefaultStoredModel(
  catalogEntries: readonly DemoCatalogEntry[],
  registeredEntries: readonly Array<{
    modelId: string;
    createdAt?: string;
    savedAtUtc?: string;
  }>,
  preferredModelId?: string | null
): DemoCatalogEntry | null;

export declare function loadCatalog(): Promise<DemoCatalogEntry[]>;

export declare function checkStoredModels(): Promise<Array<Record<string, unknown> & { modelId: string }>>;

export declare function loadDefaultStoredModel(): Promise<unknown | null>;

export declare function renderModelCards(): void;
