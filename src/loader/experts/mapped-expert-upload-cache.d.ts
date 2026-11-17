/** Upload ownership only. Keys and complete payload authentication belong to the caller. */
export interface MappedExpertUploadCacheOptions {
  device: GPUDevice;
  /** This module's share of the existing total model GPUBuffer cache budget. */
  maxBytes: number;
  /** Caller-reserved share of the existing aggregate transient staging allowance. */
  maxMappedBytes: number;
  maxEntries: number;
  /** Defaults retain ordinary admission and LRU behavior. */
  admission?: 'all' | 'repeat-read';
  /** Protected byte share of the remaining non-pinned cache budget, in [0, 1]. */
  protectedShare?: number;
  /** Bounded first-miss metadata; defaults to maxEntries. */
  maxGhostEntries?: number;
  /** Synchronous ports must retain outer charges on failure and include every allocation. */
  allocate?: (descriptor: GPUBufferDescriptor) => GPUBuffer;
  dispose?: (buffer: GPUBuffer) => void;
}
/** Borrowed mapping invalidates at publication/unmap. Return the exact completed byte count. */
export type MappedExpertFill = (destination: Uint8Array, signal: AbortSignal) => number | Promise<number>;
export interface MappedExpertCopy {
  sourceOffset: number;
  destination: GPUBuffer;
  destinationOffset: number;
  bytes: number;
}
export interface MappedExpertEntry {
  key: string; bytes: number; buffer: GPUBuffer; mapped: boolean; mapping: boolean; pinned: boolean; protected: boolean; firstConsumption: boolean; users: number;
}
export interface MappedExpertUse {
  entry: MappedExpertEntry; attempt: Promise<void> | null; failure: unknown;
}
export interface MappedExpertUploadStats {
  admission: 'all' | 'repeat-read'; protectedShare: number; maxGhostEntries: number;
  probationEntries: number; protectedEntries: number; protectedBytes: number; protectedBudgetBytes: number; ghostEntries: number;
  allocations: number; reuseAttempts: number;
  /** Completed native remaps, including ones cancelled before publication. */
  reuses: number;
  admissionSkips: number;
  /** Copy reuse excludes the first consumption after each publication, including remapped publication. */
  reuseHits: number;
  maxBytes: number; maxMappedBytes: number; maxEntries: number;
  allocatedBytes: number; mappedBytes: number; peakBytes: number; peakMappedBytes: number;
  publishedBytes: number; pinnedBytes: number; evictableBytes: number; unpublishedBytes: number;
  entries: number; pinnedEntries: number; evictableEntries: number; activeWriters: number;
  pendingMaps: number; pendingCopies: number; ownedAllocations: number;
  hits: number; misses: number; admissions: number; evictions: number; rejections: number;
  copySubmissions: number; copyBytes: number;
}
export declare class MappedExpertUploadCache {
  constructor(options: MappedExpertUploadCacheOptions);
  /** False means bounded admission is unavailable; never evicts pinned/busy entries. */
  admit(key: string, bytes: number, fill: MappedExpertFill, options?: { pinned?: boolean; signal?: AbortSignal | null }): Promise<boolean>;
  /** One owned submission. A miss returns null; its source stays pinned until actual queue settlement. */
  copyInto(key: string, copies: readonly MappedExpertCopy[], signal?: AbortSignal | null): { bytes: number; settled: Promise<void> } | null;
  /** Explicit retry of failed submitted-copy fences. */
  settle(): Promise<void>;
  getStats(): MappedExpertUploadStats;
  /** Stops admission immediately; waits writers/copies; failed retirement retains charges for retry. */
  close(): Promise<void>;
}
