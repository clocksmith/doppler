/**
 * KV Cache Types - Shared interfaces and utilities
 *
 * @module inference/kv-cache/types
 */

// ============================================================================
// Configuration
// ============================================================================

/**
 * KV Cache Configuration
 */
export interface KVCacheConfig {
  numLayers: number;
  numHeads: number;
  headDim: number;
  maxSeqLen: number;
  useGPU?: boolean;
  layout?: 'contiguous' | 'paged';
  pageSize?: number;
  kvDtype?: 'f16' | 'f32';
  /** Window size for sliding window cache */
  windowSize?: number;
}

// ============================================================================
// Layer Cache Types
// ============================================================================

/**
 * Cache entry for a single layer (contiguous layout)
 */
export interface ContiguousLayerCache {
  keys: Float32Array;
  values: Float32Array;
  keysGPU: GPUBuffer | null;
  valuesGPU: GPUBuffer | null;
  seqLen: number;
}

/**
 * Cache entry for a single layer (paged layout)
 */
export interface PagedLayerCache {
  keyPages: (Float32Array | null)[];
  valuePages: (Float32Array | null)[];
  allocatedPages: number;
  seqLen: number;
}

/**
 * Union type for layer cache entries
 */
export type LayerCache = ContiguousLayerCache | PagedLayerCache;

// ============================================================================
// Result Types
// ============================================================================

/**
 * Page location information
 */
export interface PageLocation {
  pageIdx: number;
  offset: number;
}

/**
 * KV cache get result
 */
export interface KVGetResult {
  keys: Float32Array;
  values: Float32Array;
}

/**
 * GPU buffers result
 */
export interface GPUBuffersResult {
  keysGPU: GPUBuffer;
  valuesGPU: GPUBuffer;
  seqLen: number;
}

/**
 * Memory statistics
 */
export interface MemoryStats {
  theoretical: number;
  allocated: number;
  used: number;
  efficiency: number;
  seqLen: number;
  maxSeqLen: number;
  layout: 'contiguous' | 'paged';
}

/**
 * GPU context for cache migration
 */
export interface GPUContext {
  device: GPUDevice;
}

// ============================================================================
// Type Guards
// ============================================================================

/**
 * Type guard to check if layer is contiguous
 */
export function isContiguousLayer(layer: LayerCache): layer is ContiguousLayerCache {
  return 'keys' in layer && 'values' in layer;
}

/**
 * Type guard to check if layer is paged
 */
export function isPagedLayer(layer: LayerCache): layer is PagedLayerCache {
  return 'keyPages' in layer && 'valuePages' in layer;
}

// ============================================================================
// F16 Conversion Utilities
// ============================================================================

const f32View = new Float32Array(1);
const u32View = new Uint32Array(f32View.buffer);

/**
 * Convert a single F32 value to F16 bits
 */
export function f32ToF16Bits(value: number): number {
  f32View[0] = value;
  const x = u32View[0];
  const sign = (x >> 16) & 0x8000;
  let exp = ((x >> 23) & 0xff) - 127 + 15;
  let mant = x & 0x7fffff;

  if (exp <= 0) {
    if (exp < -10) return sign;
    mant = (mant | 0x800000) >> (1 - exp);
    return sign | ((mant + 0x1000) >> 13);
  }

  if (exp >= 31) {
    return sign | 0x7c00 | (mant ? 0x200 : 0);
  }

  return sign | (exp << 10) | ((mant + 0x1000) >> 13);
}

/**
 * Convert F16 bits to F32 value
 */
export function f16ToF32Bits(h: number): number {
  const sign = (h >> 15) & 0x1;
  const exp = (h >> 10) & 0x1f;
  const mant = h & 0x3ff;

  if (exp === 0) {
    if (mant === 0) return sign ? -0 : 0;
    const f = mant / 1024 * Math.pow(2, -14);
    return sign ? -f : f;
  }
  if (exp === 31) {
    return mant ? NaN : (sign ? -Infinity : Infinity);
  }

  const f = (1 + mant / 1024) * Math.pow(2, exp - 15);
  return sign ? -f : f;
}

/**
 * Convert F32 array to F16 (Uint16Array)
 */
export function f32ToF16Array(input: Float32Array): Uint16Array {
  const out = new Uint16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    out[i] = f32ToF16Bits(input[i]);
  }
  return out;
}

/**
 * Convert F16 array to F32
 */
export function f16ToF32Array(input: Uint16Array): Float32Array {
  const out = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    out[i] = f16ToF32Bits(input[i]);
  }
  return out;
}
