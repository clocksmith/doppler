/**
 * KV Cache Module - Re-export facade
 *
 * This file re-exports from the kv-cache/ directory for backward compatibility.
 * New code should import directly from 'inference/kv-cache/index.js'.
 *
 * @module inference/kv-cache
 */

export {
  // Types
  type KVCacheConfig,
  type ContiguousLayerCache,
  type PagedLayerCache,
  type LayerCache,
  type PageLocation,
  type KVGetResult,
  type GPUBuffersResult,
  type TieredGPUBuffersResult,
  type BDPAGPUBuffersResult,
  type MemoryStats,
  type GPUContext,
  isContiguousLayer,
  isPagedLayer,
  f32ToF16Bits,
  f16ToF32Bits,
  f32ToF16Array,
  f16ToF32Array,
  // Classes
  KVCache,
  SlidingWindowKVCache,
  TieredKVCache,
  // Default
  KVCache as default,
} from './kv-cache/index.js';
