/**
 * DOPPLER Debug Module - Tensor Inspection Utilities
 *
 * Tools for inspecting GPU and CPU tensors, checking health, and comparing values.
 *
 * @module debug/tensor
 */

// ============================================================================
// Types
// ============================================================================

/**
 * Tensor statistics
 */
export interface TensorStats {
  label: string;
  shape: number[];
  size: number;
  isGPU: boolean;
  min: number;
  max: number;
  mean: number;
  std: number;
  nanCount: number;
  infCount: number;
  zeroCount: number;
  zeroPercent: string;
  first: string[];
  last: string[];
}

/**
 * Tensor comparison result
 */
export interface TensorCompareResult {
  label: string;
  match: boolean;
  maxDiff: number;
  maxDiffIdx: number;
  avgDiff: number;
  mismatchCount: number;
  mismatchPercent: string;
  error?: string;
}

/**
 * Tensor health check result
 */
export interface TensorHealthResult {
  label: string;
  healthy: boolean;
  issues: string[];
}

/**
 * Tensor inspect options
 */
export interface TensorInspectOptions {
  shape?: number[];
  maxPrint?: number;
  checkNaN?: boolean;
}

export interface TensorSnapshot {
  shape: number[];
  dtype: string;
  stats: {
    min: number;
    max: number;
    maxAbs: number;
    mean: number;
    std: number;
  };
  sample: number[];
  hasNaN: boolean;
  hasInf: boolean;
}

// ============================================================================
// Tensor Inspection Interface
// ============================================================================

/**
 * Tensor inspection utilities.
 */
export declare const tensor: {
  /**
   * Inspect a GPU or CPU tensor and log statistics.
   */
  inspect(
    buffer: GPUBuffer | Float32Array | Float64Array | Uint16Array,
    label: string,
    options?: TensorInspectOptions
  ): Promise<TensorStats | null>;

  /**
   * Compare two tensors element-wise.
   */
  compare(
    a: Float32Array,
    b: Float32Array,
    label: string,
    tolerance?: number
  ): TensorCompareResult;

  /**
   * Check tensor for common issues.
   */
  healthCheck(data: Float32Array, label: string): TensorHealthResult;
};

export declare function snapshotTensor(
  buffer: GPUBuffer,
  shape?: number[],
  dtype?: string
): Promise<TensorSnapshot>;

export declare function snapshotFromArray(
  arr: Float32Array,
  shape: number[],
  dtype?: string
): TensorSnapshot;
