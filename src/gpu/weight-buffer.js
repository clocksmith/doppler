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

/**
 * Create a weight buffer from a GPU buffer with explicit metadata.
 * @param {GPUBuffer} buffer
 * @param {import('./weight-buffer.js').WeightDtype} dtype
 * @param {import('./weight-buffer.js').WeightLayout} layout
 * @param {number[]} shape
 * @param {string} [label]
 * @returns {import('./weight-buffer.js').WeightBuffer}
 */
export function createWeightBuffer(
  buffer,
  dtype,
  layout,
  shape,
  label
) {
  return {
    buffer,
    dtype,
    layout,
    shape: Object.freeze([...shape]),
    label,
  };
}

/**
 * Create a CPU-resident weight buffer with explicit metadata.
 * @param {Float32Array} data
 * @param {import('./weight-buffer.js').WeightDtype} dtype
 * @param {import('./weight-buffer.js').WeightLayout} layout
 * @param {number[]} shape
 * @param {string} [label]
 * @returns {import('./weight-buffer.js').CpuWeightBuffer}
 */
export function createCpuWeightBuffer(
  data,
  dtype,
  layout,
  shape,
  label
) {
  return {
    data,
    dtype,
    layout,
    shape: Object.freeze([...shape]),
    label,
  };
}

/**
 * Check if weight is stored in column-major (pre-transposed) format.
 * Column-major weights use transposeB=false in matmul.
 * @param {import('./weight-buffer.js').WeightBuffer} weight
 * @returns {boolean}
 */
export function isColumnMajor(weight) {
  return weight.layout === 'column';
}

/**
 * Check if weight buffer is a specific type for type guards.
 * @param {unknown} value
 * @returns {value is import('./weight-buffer.js').WeightBuffer}
 */
export function isWeightBuffer(value) {
  return (
    typeof value === 'object' &&
    value !== null &&
    'buffer' in value &&
    'dtype' in value &&
    'layout' in value &&
    'shape' in value
  );
}

/**
 * Check if value is a CPU-resident weight buffer.
 * @param {unknown} value
 * @returns {value is import('./weight-buffer.js').CpuWeightBuffer}
 */
export function isCpuWeightBuffer(value) {
  return (
    typeof value === 'object' &&
    value !== null &&
    'data' in value &&
    'dtype' in value &&
    'layout' in value &&
    'shape' in value
  );
}

/**
 * Extract the raw GPUBuffer from either a WeightBuffer or raw GPUBuffer.
 * Used for backwards compatibility during migration.
 * @param {GPUBuffer | import('./weight-buffer.js').WeightBuffer} weight
 * @returns {GPUBuffer}
 */
export function getBuffer(weight) {
  return isWeightBuffer(weight) ? weight.buffer : weight;
}

/**
 * Get layout from WeightBuffer, or null for raw GPUBuffer.
 * Used for auto-resolving transposeB in matmul.
 * @param {GPUBuffer | import('./weight-buffer.js').WeightBuffer} weight
 * @returns {import('./weight-buffer.js').WeightLayout | null}
 */
export function getLayout(weight) {
  return isWeightBuffer(weight) ? weight.layout : null;
}

/**
 * Get dtype from WeightBuffer, or null for raw GPUBuffer.
 * @param {GPUBuffer | import('./weight-buffer.js').WeightBuffer} weight
 * @returns {import('./weight-buffer.js').WeightDtype | null}
 */
export function getWeightDtype(weight) {
  return isWeightBuffer(weight) ? weight.dtype : null;
}
