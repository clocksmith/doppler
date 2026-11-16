/** Dequantized MXFP4 expert weights owned by one model's loader; cleared on unload. */
export declare class ExpertDequantCache {
  get(layerIdx: number, expertIdx: number, outputDtype: 'f16' | 'f32'): { gateUp: GPUBuffer; down: GPUBuffer } | undefined;
  set(layerIdx: number, expertIdx: number, outputDtype: 'f16' | 'f32', gateUp: GPUBuffer, down: GPUBuffer): void;
  clear(): void;
  /** Overrides runtime.inference.moe.cache.dequantCacheMaxEntries; null restores it. */
  setMaxEntries(maxEntries: number | null): void;
  getStats(): { hits: number; misses: number; size: number; maxEntries: number };
}
