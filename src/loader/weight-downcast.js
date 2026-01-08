/**
 * Weight Downcast - F32 to F16 weight conversion utility.
 *
 * Provides a unified utility for downcasting F32 weights to F16 when
 * the GPU supports shader-f16. Used by layer loading, expert loading,
 * embedding loading, and final weights loading.
 *
 * @module loader/weight-downcast
 */

import { getKernelCapabilities } from '../gpu/device.js';
import { createTensor } from '../gpu/tensor.js';
import { castF32ToF16 } from '../gpu/kernel-selector.js';
import { releaseBuffer } from '../gpu/buffer-pool.js';
import {
  createWeightBuffer,
  isWeightBuffer,
  getWeightDtype,
  getLayout,
} from '../gpu/weight-buffer.js';
import { trace as debugTrace } from '../debug/index.js';

// ============================================================================
// Main Downcast Function
// ============================================================================

/**
 * Attempt to downcast a weight buffer from F32 to F16.
 *
 * @param {GPUBuffer | import('../gpu/weight-buffer.js').WeightBuffer | null} buf - Input buffer (GPUBuffer or WeightBuffer)
 * @param {import('./weight-downcast.js').DowncastOptions} options - Downcast options
 * @returns {Promise<import('./weight-downcast.js').DowncastResult | null>} Downcast result, or null if input is null/unsupported
 */
export async function maybeDowncastToF16(buf, options) {
  if (!buf) return null;

  const caps = getKernelCapabilities();
  if (!caps.hasF16) {
    // No F16 support - return as-is
    return {
      buffer: buf,
      wasDowncast: false,
      newBuffer: null,
    };
  }

  if (options.keepF32) {
    const layerStr = options.layerIdx !== undefined ? `Layer ${options.layerIdx}` : '';
    debugTrace.loader(`${layerStr} keeping ${options.label} in f32 (keepF32Weights=true)`);
    return {
      buffer: buf,
      wasDowncast: false,
      newBuffer: null,
    };
  }

  // Handle WeightBuffer
  if (isWeightBuffer(buf)) {
    return downcastWeightBuffer(buf, options);
  }

  // Handle raw GPUBuffer
  if (buf instanceof GPUBuffer) {
    return downcastGPUBuffer(buf, options);
  }

  // Unsupported type
  return {
    buffer: buf,
    wasDowncast: false,
    newBuffer: null,
  };
}

// ============================================================================
// Internal Helpers
// ============================================================================

/**
 * Downcast a WeightBuffer from F32 to F16.
 * @param {import('../gpu/weight-buffer.js').WeightBuffer} buf
 * @param {import('./weight-downcast.js').DowncastOptions} options
 * @returns {Promise<import('./weight-downcast.js').DowncastResult>}
 */
async function downcastWeightBuffer(buf, options) {
  const dtype = getWeightDtype(buf);
  if (dtype !== 'f32') {
    // Already F16 or other dtype
    return {
      buffer: buf,
      wasDowncast: false,
      newBuffer: null,
    };
  }

  const elems = buf.buffer.size / 4;
  const wasColumnMajor = getLayout(buf) === 'column';
  const layerStr = options.layerIdx !== undefined ? `Layer ${options.layerIdx}` : '';

  debugTrace.loader(
    `${layerStr} downcasting WeightBuffer ${options.label}: ` +
    `bufSize=${buf.buffer.size}, elems=${elems}, columnMajor=${wasColumnMajor}`
  );

  try {
    const inputTensor = createTensor(buf.buffer, 'f32', [elems], `${options.label}_f32`);
    const f16Tensor = await castF32ToF16(inputTensor);

    // Create new WeightBuffer with f16 dtype, preserving layout
    const layout = options.layout ?? (wasColumnMajor ? 'column' : 'row');
    const shape = options.shape ?? /** @type {number[]} */ (buf.shape);
    const newWeightBuffer = createWeightBuffer(
      f16Tensor.buffer,
      'f16',
      layout,
      shape,
      buf.label ?? options.label
    );

    debugTrace.loader(`${layerStr} ${options.label} downcast result: f16Size=${f16Tensor.buffer.size}`);

    // Release old buffer
    releaseBuffer(buf.buffer);

    return {
      buffer: newWeightBuffer,
      wasDowncast: true,
      newBuffer: f16Tensor.buffer,
    };
  } catch (e) {
    debugTrace.loader(`Failed to downcast ${options.label} to f16: ${/** @type {Error} */ (e).message}`);
    return {
      buffer: buf,
      wasDowncast: false,
      newBuffer: null,
    };
  }
}

/**
 * Downcast a raw GPUBuffer from F32 to F16.
 * @param {GPUBuffer} buf
 * @param {import('./weight-downcast.js').DowncastOptions} options
 * @returns {Promise<import('./weight-downcast.js').DowncastResult>}
 */
async function downcastGPUBuffer(buf, options) {
  const dtype = getWeightDtype(buf) || 'f32';
  if (dtype !== 'f32') {
    // Already F16 or other dtype
    return {
      buffer: buf,
      wasDowncast: false,
      newBuffer: null,
    };
  }

  const elems = buf.size / 4;
  const wasColumnMajor = getLayout(buf) === 'column';
  const layerStr = options.layerIdx !== undefined ? `Layer ${options.layerIdx}` : '';

  debugTrace.loader(
    `${layerStr} downcasting ${options.label}: ` +
    `bufSize=${buf.size}, elems=${elems}, expectedF16=${elems * 2}, columnMajor=${wasColumnMajor}`
  );

  try {
    const inputTensor = createTensor(buf, 'f32', [elems], `${options.label}_f32`);
    const f16Tensor = await castF32ToF16(inputTensor);

    // Create WeightBuffer with f16 dtype, preserving layout
    const layout = options.layout ?? (wasColumnMajor ? 'column' : 'row');
    const shape = options.shape ?? [elems];
    const newWeightBuffer = createWeightBuffer(
      f16Tensor.buffer,
      'f16',
      layout,
      shape,
      options.label
    );

    debugTrace.loader(`${layerStr} ${options.label} downcast result: f16Size=${f16Tensor.buffer.size}`);

    // Release old buffer
    releaseBuffer(buf);

    return {
      buffer: newWeightBuffer,
      wasDowncast: true,
      newBuffer: f16Tensor.buffer,
    };
  } catch (e) {
    debugTrace.loader(`Failed to downcast ${options.label} to f16: ${/** @type {Error} */ (e).message}`);
    return {
      buffer: buf,
      wasDowncast: false,
      newBuffer: null,
    };
  }
}

// ============================================================================
// Batch Downcast Helper
// ============================================================================

/**
 * Downcast multiple weight buffers, tracking new GPU buffers.
 *
 * @template {Record<string, GPUBuffer | import('../gpu/weight-buffer.js').WeightBuffer | null>} T
 * @param {T} weights - Record of weight buffers to downcast
 * @param {(keyof T)[]} keys - Keys to downcast
 * @param {Omit<import('./weight-downcast.js').DowncastOptions, 'label'>} options - Base options (label will be set per key)
 * @param {Set<GPUBuffer>} gpuBuffers - Set to track new GPU buffers
 * @returns {Promise<void>}
 */
export async function batchDowncastWeights(weights, keys, options, gpuBuffers) {
  for (const key of keys) {
    const buf = weights[key];
    if (!buf) continue;

    const result = await maybeDowncastToF16(/** @type {GPUBuffer | import('../gpu/weight-buffer.js').WeightBuffer} */ (buf), {
      ...options,
      label: String(key),
    });

    if (result?.wasDowncast) {
      /** @type {Record<string, unknown>} */ (weights)[/** @type {string} */ (key)] = result.buffer;
      if (result.newBuffer) {
        gpuBuffers.add(result.newBuffer);
      }
    }
  }
}
