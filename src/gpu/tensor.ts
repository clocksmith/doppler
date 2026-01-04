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
): Tensor {
  return {
    buffer,
    dtype,
    shape: Object.freeze([...shape]),
    label,
  };
}

/**
 * Assert tensor has expected dtype, throw if mismatch.
 */
export function assertDtype(
  tensor: Tensor,
  expected: TensorDtype,
  operation: string
): void {
  if (tensor.dtype !== expected) {
    throw new Error(
      `${operation}: expected ${expected} tensor, got ${tensor.dtype}` +
      (tensor.label ? ` (${tensor.label})` : '')
    );
  }
}

/**
 * Assert tensor has expected shape, throw if mismatch.
 */
export function assertShape(
  tensor: Tensor,
  expected: readonly number[],
  operation: string
): void {
  if (tensor.shape.length !== expected.length) {
    throw new Error(
      `${operation}: expected ${expected.length}D tensor, got ${tensor.shape.length}D` +
      (tensor.label ? ` (${tensor.label})` : '')
    );
  }
  for (let i = 0; i < expected.length; i++) {
    if (expected[i] !== -1 && tensor.shape[i] !== expected[i]) {
      throw new Error(
        `${operation}: shape mismatch at dim ${i}: expected ${expected[i]}, got ${tensor.shape[i]}` +
        (tensor.label ? ` (${tensor.label})` : '')
      );
    }
  }
}

/**
 * Get bytes per element for dtype.
 */
export function dtypeBytes(dtype: TensorDtype): number {
  return dtype === 'f16' ? 2 : 4;
}

/**
 * Compute total byte size for a tensor.
 */
export function tensorBytes(shape: readonly number[], dtype: TensorDtype): number {
  return shape.reduce((a, b) => a * b, 1) * dtypeBytes(dtype);
}

/**
 * Check if two tensors have compatible dtypes for an operation.
 */
export function dtypesMatch(a: Tensor, b: Tensor): boolean {
  return a.dtype === b.dtype;
}

/**
 * Determine output dtype for a binary operation.
 * F16 only if both inputs are F16.
 */
export function inferOutputDtype(a: Tensor, b: Tensor): TensorDtype {
  return (a.dtype === 'f16' && b.dtype === 'f16') ? 'f16' : 'f32';
}
