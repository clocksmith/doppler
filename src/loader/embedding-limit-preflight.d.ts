export const MAX_SPLIT4_EMBEDDING_SECTIONS: 4;
export const MAX_SPLIT8_EMBEDDING_SECTIONS: 8;
export const MAX_SPLIT_EMBEDDING_SECTIONS: 8;
export const SPLIT_EMBEDDING_STORAGE_BUFFER_OVERHEAD: 2;

export function getEmbeddingFloatDtype(location: unknown): string;

export function expectsSplitGpuEmbeddingKernel(embeddingKernel: unknown): boolean;

export function getSplitGpuEmbeddingKernelSectionCount(embeddingKernel: unknown): number;

export function getSplitGpuEmbeddingRequiredStorageBuffers(sectionCount: number): number;

export function getMaxSplitGpuEmbeddingSectionsForDevice(
  embeddingKernel?: unknown,
  device?: unknown
): number;

export function createGpuResidentEmbeddingLimitError(options: {
  name: string;
  location: unknown;
  embeddingKernel?: unknown;
}): Error | null;

export function resolveManifestGpuResidentEmbeddingLimitError(
  manifest: unknown,
  options?: {
    storageManifest?: unknown;
    runtimeConfig?: unknown;
  }
): Error | null;
