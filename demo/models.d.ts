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

export declare function assertDemoExecutionManifestSupported(
  entry: DemoCatalogEntry,
  manifest: Record<string, unknown>,
  capabilities: Record<string, unknown>
): void;

export declare function selectDemoExecutionEntryForCapabilities(
  entry: DemoCatalogEntry,
  manifestByModelId: Map<string, Record<string, unknown>>,
  capabilities: Record<string, unknown>
): DemoCatalogEntry;

export declare function findPrimaryForWeightPack(
  catalogEntries: readonly DemoCatalogEntry[],
  weightPackId: string
): DemoCatalogEntry | null;

export declare function assertWeightsRefPrimaryAvailable(
  entry: DemoCatalogEntry,
  catalogEntries: readonly DemoCatalogEntry[],
  storedModelIds: Set<string>
): void;

export declare function buildRemoveConfirmText(entry: DemoCatalogEntry): string;

export declare function buildModelCardDetail(
  entry: DemoCatalogEntry,
  status: string | null | undefined
): string;

export declare function findRegisteredSiblingsOf(
  primaryEntry: DemoCatalogEntry,
  catalogEntries: readonly DemoCatalogEntry[],
  storedModelIds: Set<string>
): DemoCatalogEntry[];

export declare function setModelCallbacks(callbacks: {
  onLoaded?: ((pipeline: unknown, modelId: string) => void) | null;
  onDownloadProgress?: ((progress: unknown) => void) | null;
}): void;

export declare function buildLocalModelBaseUrl(
  modelId: string,
  origin?: string | null
): string;

export declare function selectDemoCatalogEntries(
  models: readonly DemoCatalogEntry[],
  options?: Record<string, unknown>
): DemoCatalogEntry[];

export declare function buildModelSourceCandidates(entry: DemoCatalogEntry): string[];

export declare function loadCatalog(): Promise<DemoCatalogEntry[]>;

export declare function checkStoredModels(): Promise<void>;

export declare function renderModelCards(): void;

export declare function patchManifestCompat(manifest: Record<string, unknown>): Record<string, unknown>;
