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
export function isContiguousLayer(layer: LayerCache): layer is ContiguousLayerCache;

/**
 * Type guard to check if layer is paged
 */
export function isPagedLayer(layer: LayerCache): layer is PagedLayerCache;

// ============================================================================
// F16 Conversion Utilities
// ============================================================================

/**
 * Convert a single F32 value to F16 bits
 */
export function f32ToF16Bits(value: number): number;

/**
 * Convert F16 bits to F32 value
 */
export function f16ToF32Bits(h: number): number;

/**
 * Convert F32 array to F16 (Uint16Array)
 */
export function f32ToF16Array(input: Float32Array): Uint16Array;

/**
 * Convert F16 array to F32
 */
export function f16ToF32Array(input: Uint16Array): Float32Array;
