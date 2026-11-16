import { getRuntimeConfig } from '../../config/runtime.js';
import { releaseBuffer } from '../../memory/buffer-pool.js';

// Dequantized MXFP4 expert weights for one loaded model. The model's loader owns
// the instance and clears it on unload, so a hit always comes from this model's
// own source weights on its own device. Map insertion order is the LRU order.
export class ExpertDequantCache {
  /** @type {Map<string, { gateUp: GPUBuffer, down: GPUBuffer }>} */
  #entries = new Map();
  /** @type {number | null} */
  #maxEntriesOverride = null;
  #hits = 0;
  #misses = 0;

  #maxEntries() {
    const moe = /** @type {{ cache?: { dequantCacheMaxEntries?: unknown } }} */ (getRuntimeConfig().inference.moe);
    const maxEntries = this.#maxEntriesOverride ?? moe.cache?.dequantCacheMaxEntries;
    if (typeof maxEntries !== 'number' || !Number.isSafeInteger(maxEntries) || maxEntries < 1) {
      throw new Error(`runtime.inference.moe.cache.dequantCacheMaxEntries must be a positive integer, got ${String(maxEntries)}.`);
    }
    return maxEntries;
  }

  /** @type {import('./expert-dequant-cache.js').ExpertDequantCache['get']} */
  get(layerIdx, expertIdx, outputDtype) {
    const key = `${layerIdx}_${expertIdx}_${outputDtype}`;
    const entry = this.#entries.get(key);
    if (!entry) return undefined;
    this.#entries.delete(key);
    this.#entries.set(key, entry);
    this.#hits += 1;
    return entry;
  }

  /** @type {import('./expert-dequant-cache.js').ExpertDequantCache['set']} */
  set(layerIdx, expertIdx, outputDtype, gateUp, down) {
    const key = `${layerIdx}_${expertIdx}_${outputDtype}`;
    this.#misses += 1;
    const replaced = this.#entries.get(key);
    if (replaced) {
      this.#entries.delete(key);
      release(replaced);
    }
    const maxEntries = this.#maxEntries();
    while (this.#entries.size >= maxEntries) {
      const oldest = this.#entries.entries().next().value;
      if (!oldest) break;
      this.#entries.delete(oldest[0]);
      release(oldest[1]);
    }
    this.#entries.set(key, { gateUp, down });
  }

  clear() {
    for (const entry of this.#entries.values()) release(entry);
    this.#entries.clear();
    this.#hits = 0;
    this.#misses = 0;
  }

  /** @param {number | null} maxEntries */
  setMaxEntries(maxEntries) {
    this.#maxEntriesOverride = maxEntries;
  }

  getStats() {
    return { hits: this.#hits, misses: this.#misses, size: this.#entries.size, maxEntries: this.#maxEntries() };
  }
}

/** @param {{ gateUp: GPUBuffer, down: GPUBuffer }} entry */
function release(entry) {
  releaseBuffer(entry.gateUp);
  releaseBuffer(entry.down);
}
