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
  useGPU: boolean;
  layout: 'contiguous' | 'contiguous_quantized' | 'paged' | 'tiered' | 'bdpa' | 'bdpa_paged';
  pageSize: number;
  kvDtype: 'f16' | 'f32';
  /** Window size for sliding window cache */
  windowSize?: number;
  /** Tiered KV cache settings (required when layout = 'tiered') */
  tiering?: {
    mode: 'off' | 'fp16' | 'int8' | 'int4' | 'turboquant' | 'turboquant_prod';
    hotWindow: number;
    coldPageSize: number;
    coldDtype: 'f16' | 'f32';
    compression: {
      mode: 'none' | 'int8' | 'int4' | 'turboquant' | 'turboquant_prod';
      blockSize: number;
      bitWidth?: number;
      prodMode?: boolean;
    };
    gating: { mode: 'auto' | 'force_on' | 'force_off'; minAluBwRatio: number };
  };
  quantization?: {
    mode: 'none' | 'turboquant' | 'turboquant_prod';
    bitWidth?: number;
    prodMode?: boolean;
  };
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
  keysGPU?: GPUBuffer | null;
  valuesGPU?: GPUBuffer | null;
  pageTable?: Uint32Array | null;
  pageTableGPU?: GPUBuffer | null;
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
 * GPU buffers result (Contiguous or Paged)
 */
export interface StandardGPUBuffersResult {
  keysGPU: GPUBuffer;
  valuesGPU: GPUBuffer;
  seqLen: number;
  layout?: 'contiguous' | 'ring' | 'paged';
  pageTableGPU?: GPUBuffer;
  pageSize?: number;
}

export interface TieredGPUBuffersResult {
  layout: 'tiered';
  seqLen: number;
  hotKeysGPU: GPUBuffer;
  hotValuesGPU: GPUBuffer;
  hotSeqLen: number;
  hotStart: number;
  hotWindow: number;
  coldKeysGPU: GPUBuffer;
  coldValuesGPU: GPUBuffer;
  coldScalesKGPU?: GPUBuffer;
  coldScalesVGPU?: GPUBuffer;
  coldSeqLen: number;
  coldPageTableGPU?: GPUBuffer;
  coldPageSize?: number;
  coldPackedStride?: number;
  coldQuantMode?: 'none' | 'int8' | 'int4' | 'turboquant' | 'turboquant_prod';
  rotationMatrixBuffer?: GPUBuffer;
  codebookCentroidsBuffer?: GPUBuffer;
  residualKGPU?: GPUBuffer;
  residualVGPU?: GPUBuffer;
  residualNormsKGPU?: GPUBuffer;
  residualNormsVGPU?: GPUBuffer;
  qjlMatrixBuffer?: GPUBuffer;
  residualPackedStride?: number;
}

export interface QuantizedGPUBuffersResult {
  layout: 'contiguous_quantized';
  seqLen: number;
  quantMode: 'turboquant' | 'turboquant_prod';
  prodMode: boolean;
  packedStride: number;
  scalesKGPU: GPUBuffer;
  scalesVGPU: GPUBuffer;
  rotationMatrixBuffer: GPUBuffer;
  codebookCentroidsBuffer: GPUBuffer;
  keysPackedGPU: GPUBuffer;
  valuesPackedGPU: GPUBuffer;
  residualKGPU?: GPUBuffer;
  residualVGPU?: GPUBuffer;
  residualNormsKGPU?: GPUBuffer;
  residualNormsVGPU?: GPUBuffer;
  residualPackedStride?: number;
  qjlMatrixBuffer?: GPUBuffer;
}

export interface BDPAGPUBuffersResult {
  layout: 'bdpa';
  seqLen: number;
  basisGPU: { k: GPUBuffer; v: GPUBuffer };
  pagedGPU: { k: GPUBuffer; v: GPUBuffer };
  indexGPU: GPUBuffer;
  numBasisVectors: number;
  pageSize: number;
}

export type GPUBuffersResult =
  | StandardGPUBuffersResult
  | TieredGPUBuffersResult
  | QuantizedGPUBuffersResult
  | BDPAGPUBuffersResult;

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
  layout: 'contiguous' | 'contiguous_quantized' | 'paged' | 'tiered' | 'bdpa_paged';
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
