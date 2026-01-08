/**
 * Tensor Abstraction
 *
 * Wraps GPUBuffer with explicit dtype and shape metadata.
 * Ensures dtype flows through the pipeline structurally rather than
 * being tracked in a separate WeakMap.
 */

/**
 * Create a tensor from a buffer with explicit dtype.
 * @param {GPUBuffer} buffer
 * @param {import('./tensor.js').TensorDtype} dtype
 * @param {number[]} shape
 * @param {string} [label]
 * @returns {import('./tensor.js').Tensor}
 */
export function createTensor(
  buffer,
  dtype,
  shape,
  label
) {
  return {
    buffer,
    dtype,
    shape: Object.freeze([...shape]),
    label,
  };
}

/**
 * Assert tensor has expected dtype, throw if mismatch.
 * @param {import('./tensor.js').Tensor} tensor
 * @param {import('./tensor.js').TensorDtype} expected
 * @param {string} operation
 * @returns {void}
 */
export function assertDtype(
  tensor,
  expected,
  operation
) {
  if (tensor.dtype !== expected) {
    throw new Error(
      `${operation}: expected ${expected} tensor, got ${tensor.dtype}` +
      (tensor.label ? ` (${tensor.label})` : '')
    );
  }
}

/**
 * Assert tensor has expected shape, throw if mismatch.
 * @param {import('./tensor.js').Tensor} tensor
 * @param {readonly number[]} expected
 * @param {string} operation
 * @returns {void}
 */
export function assertShape(
  tensor,
  expected,
  operation
) {
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
 * @param {import('./tensor.js').TensorDtype} dtype
 * @returns {number}
 */
export function dtypeBytes(dtype) {
  return dtype === 'f16' ? 2 : 4;
}

/**
 * Compute total byte size for a tensor.
 * @param {readonly number[]} shape
 * @param {import('./tensor.js').TensorDtype} dtype
 * @returns {number}
 */
export function tensorBytes(shape, dtype) {
  return shape.reduce((a, b) => a * b, 1) * dtypeBytes(dtype);
}

/**
 * Check if two tensors have compatible dtypes for an operation.
 * @param {import('./tensor.js').Tensor} a
 * @param {import('./tensor.js').Tensor} b
 * @returns {boolean}
 */
export function dtypesMatch(a, b) {
  return a.dtype === b.dtype;
}

/**
 * Determine output dtype for a binary operation.
 * F16 only if both inputs are F16.
 * @param {import('./tensor.js').Tensor} a
 * @param {import('./tensor.js').Tensor} b
 * @returns {import('./tensor.js').TensorDtype}
 */
export function inferOutputDtype(a, b) {
  return (a.dtype === 'f16' && b.dtype === 'f16') ? 'f16' : 'f32';
}
