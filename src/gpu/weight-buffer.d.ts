/**
 * Weight Buffer Abstraction
 *
 * Wraps GPUBuffer with weight-specific metadata (dtype, layout).
 * Parallel to Tensor but for weights which have:
 * - Quantized dtypes (q4k, q8, bf16)
 * - Layout metadata (row/column for transposeB)
 *
 * Use Tensor for activations (f16/f32 flowing through pipeline).
 * Use WeightBuffer for model weights (static, may be quantized).
 */

export type WeightDtype = 'f16' | 'f32' | 'bf16' | 'q4k' | 'q8';
export type WeightLayout = 'row' | 'column';

/**
 * CPU-resident weight buffer with layout metadata.
 * Used for oversized weights that cannot be bound as a single GPU buffer.
 */
export interface CpuWeightBuffer {
  readonly data: Float32Array;
  readonly dtype: WeightDtype;
  readonly layout: WeightLayout;
  readonly shape: readonly number[];
  readonly label?: string;
}

/**
 * A weight buffer with explicit dtype and layout.
 * Use this instead of raw GPUBuffer for weight matrices.
 */
export interface WeightBuffer {
  readonly buffer: GPUBuffer;
  readonly dtype: WeightDtype;
  readonly layout: WeightLayout;
  readonly shape: readonly number[];
  readonly label?: string;
}

/**
 * Tensor-like buffer with dtype + shape metadata.
 * Used by matmul when activations are passed in place of weights.
 */
export interface TensorLike {
  readonly buffer: GPUBuffer;
  readonly dtype: 'f16' | 'f32';
  readonly shape: readonly number[];
  readonly label?: string;
}

/**
 * Create a weight buffer from a GPU buffer with explicit metadata.
 */
export function createWeightBuffer(
  buffer: GPUBuffer,
  dtype: WeightDtype,
  layout: WeightLayout,
  shape: number[],
  label?: string
): WeightBuffer;

/**
 * Create a CPU-resident weight buffer with explicit metadata.
 */
export function createCpuWeightBuffer(
  data: Float32Array,
  dtype: WeightDtype,
  layout: WeightLayout,
  shape: number[],
  label?: string
): CpuWeightBuffer;

/**
 * Check if weight is stored in column-major (pre-transposed) format.
 * Column-major weights use transposeB=false in matmul.
 */
export function isColumnMajor(weight: WeightBuffer): boolean;

/**
 * Check if weight buffer is a specific type for type guards.
 */
export function isWeightBuffer(value: unknown): value is WeightBuffer;

/**
 * Check if value is a CPU-resident weight buffer.
 */
export function isCpuWeightBuffer(value: unknown): value is CpuWeightBuffer;

/**
 * Extract the raw GPUBuffer from either a WeightBuffer or raw GPUBuffer.
 * Used for backwards compatibility during migration.
 */
export function getBuffer(weight: GPUBuffer | WeightBuffer | TensorLike): GPUBuffer;

/**
 * Get layout from WeightBuffer, or null for raw GPUBuffer.
 * Used for auto-resolving transposeB in matmul.
 */
export function getLayout(weight: GPUBuffer | WeightBuffer | TensorLike): WeightLayout | null;

/**
 * Get dtype from WeightBuffer, or null for raw GPUBuffer.
 */
export function getWeightDtype(weight: GPUBuffer | WeightBuffer | TensorLike): WeightDtype | TensorLike['dtype'] | null;
