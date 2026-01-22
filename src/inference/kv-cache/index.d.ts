/**
 * KV Cache Module - Re-exports
 *
 * @module inference/kv-cache
 */

// Types
export {
  type KVCacheConfig,
  type ContiguousLayerCache,
  type PagedLayerCache,
  type LayerCache,
  type PageLocation,
  type KVGetResult,
  type GPUBuffersResult,
  type TieredGPUBuffersResult,
  type MemoryStats,
  type GPUContext,
  isContiguousLayer,
  isPagedLayer,
  f32ToF16Bits,
  f16ToF32Bits,
  f32ToF16Array,
  f16ToF32Array,
} from './types.js';

// Classes
export { KVCache } from './base.js';
export { SlidingWindowKVCache } from './sliding-window.js';
export { TieredKVCache } from './tiered.js';

// Default export for backward compatibility
export { KVCache as default } from './base.js';
