/**
 * Tensor Abstraction
 *
 * Wraps GPUBuffer with explicit dtype and shape metadata.
 * Ensures dtype flows through the pipeline structurally rather than
 * being tracked in a separate WeakMap.
 */

export type TensorDtype = 'f16' | 'f32';
export interface TensorRepresentation {
  readonly precision?: 'two-f32';
  readonly components?: 2;
  readonly elementStrideBytes?: 8;
}


/**
 * A tensor with explicit dtype and shape.
 * Use this instead of raw GPUBuffer for dtype-sensitive operations.
 */
export interface Tensor extends TensorRepresentation {
  readonly buffer: GPUBuffer;
  readonly dtype: TensorDtype;
  readonly shape: readonly number[];
  readonly label?: string;
  outputGateFused?: boolean;
}

/**
 * Create a tensor from a buffer with explicit dtype.
 */
export function createTensor(
  buffer: GPUBuffer,
  dtype: TensorDtype,
  shape: readonly number[],
  label?: string,
  representation?: TensorRepresentation
): Tensor;

/**
 * Get bytes per element for dtype.
 */
export function dtypeBytes(dtype: TensorDtype): number;

/**
 * Compute total byte size for a tensor.
 */
export function tensorBytes(shape: readonly number[], dtype: TensorDtype, representation?: TensorRepresentation): number;
/** Bytes per logical element; metadata must be explicit and internally consistent. */
export function tensorElementBytes(dtype: TensorDtype, representation?: TensorRepresentation): number;

/**
 * Determine output dtype for a binary operation.
 * F16 only if both inputs are F16.
 */
export function inferOutputDtype(a: Tensor, b: Tensor): TensorDtype;
