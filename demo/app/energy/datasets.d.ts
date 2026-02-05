export function loadVliwDataset(datasetId: string): Promise<Record<string, unknown>>;
export function computeDagHash(dataset: Record<string, unknown>): Promise<string>;
export function buildVliwDatasetFromSpecInput(specInput: Record<string, unknown>, cacheKey?: string): Promise<Record<string, unknown>>;
export function sliceVliwDataset(dataset: Record<string, unknown>, bundleLimit?: number): Record<string, unknown>;
