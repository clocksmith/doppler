import type { KVCache } from './base.js';
export function resolveKVCacheLayerCount(numLayers: number, layerRange: readonly number[] | null): number;
/** Consume a cache whose allocated layers are local; expose original model indices. */
export function scopeKVCacheLayerRange<T extends KVCache>(cache: T, firstLayer: number): T;
