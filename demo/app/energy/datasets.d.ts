export function loadVliwDataset(datasetId: string): Promise<Record<string, unknown>>;
export function computeDagHash(dataset: Record<string, unknown>): Promise<string>;
export function applyWorkloadSpec(specInput: Record<string, unknown>, workloadSpec?: Record<string, unknown>): Record<string, unknown>;
export function buildVliwDatasetFromSpecInput(
  specInput: Record<string, unknown>,
  cacheKey?: string,
  options?: Record<string, unknown>,
): Promise<Record<string, unknown>>;
export function sliceVliwDataset(dataset: Record<string, unknown>, bundleLimit?: number): Record<string, unknown>;
