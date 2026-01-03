/**
 * Tensor Abstraction
 *
 * Wraps GPUBuffer with explicit dtype and shape metadata.
 * Ensures dtype flows through the pipeline structurally rather than
 * being tracked in a separate WeakMap.
 */

export type TensorDtype = 'f16' | 'f32';

/**
 * A tensor with explicit dtype and shape.
 * Use this instead of raw GPUBuffer for dtype-sensitive operations.
 */
export interface Tensor {
  readonly buffer: GPUBuffer;
  readonly dtype: TensorDtype;
  readonly shape: readonly number[];
  readonly label?: string;
}

/**
 * Create a tensor from a buffer with explicit dtype.
 */
export function createTensor(
  buffer: GPUBuffer,
  dtype: TensorDtype,
  shape: number[],
  label?: string
): Tensor;

/**
 * Assert tensor has expected dtype, throw if mismatch.
 */
export function assertDtype(
  tensor: Tensor,
  expected: TensorDtype,
  operation: string
): void;

/**
 * Assert tensor has expected shape, throw if mismatch.
 */
export function assertShape(
  tensor: Tensor,
  expected: readonly number[],
  operation: string
): void;

/**
 * Get bytes per element for dtype.
 */
export function dtypeBytes(dtype: TensorDtype): number;

/**
 * Compute total byte size for a tensor.
 */
export function tensorBytes(shape: readonly number[], dtype: TensorDtype): number;

/**
 * Check if two tensors have compatible dtypes for an operation.
 */
export function dtypesMatch(a: Tensor, b: Tensor): boolean;

/**
 * Determine output dtype for a binary operation.
 * F16 only if both inputs are F16.
 */
export function inferOutputDtype(a: Tensor, b: Tensor): TensorDtype;
