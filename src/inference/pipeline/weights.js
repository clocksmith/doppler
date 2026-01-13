/**
 * Weight buffer management utilities.
 *
 * This module handles:
 * - Creating GPU buffers from CPU weight data
 * - Handling RMSNorm weight buffers (offset is applied at runtime)
 * - Type guards for layer weight structures
 * - Buffer lifecycle management
 *
 * @module inference/pipeline/weights
 */

import { getDevice } from '../../gpu/device.js';
import { acquireBuffer } from '../../gpu/buffer-pool.js';
import { log } from '../../debug/index.js';
import { isWeightBuffer, isCpuWeightBuffer } from '../../gpu/weight-buffer.js';

// ============================================================================
// Type Guards
// ============================================================================

/**
 * Type guard to check if a weight value is a LayerWeights object.
 *
 * Distinguishes between:
 * - LayerWeights objects (have qProj, kProj, etc.)
 * - Float32Array (ArrayBuffer view)
 * - GPUBuffer (has getMappedRange method)
 *
 * @param {unknown} value - Value to check
 * @returns {value is import('./types.js').LayerWeights}
 */
export function isLayerWeights(value) {
  return value !== null && typeof value === 'object' && !ArrayBuffer.isView(value) && !('getMappedRange' in /** @type {object} */ (value)) && !isWeightBuffer(value) && !isCpuWeightBuffer(value);
}

/**
 * Get layer weights from weights map with type narrowing.
 *
 * @param {Map<string, import('./types.js').LayerWeights | Float32Array | GPUBuffer>} weights - Map of weight names to weight data
 * @param {string} key - Key to look up (e.g., "layer_0")
 * @returns {import('./types.js').LayerWeights | null}
 */
export function getLayerWeights(weights, key) {
  const value = weights.get(key);
  if (value && isLayerWeights(value)) return value;
  return null;
}

// ============================================================================
// Weight Buffer Creation
// ============================================================================

/**
 * Get or create GPU buffer for a weight tensor.
 *
 * If the weight is a WeightBuffer, returns it unchanged (preserves dtype/layout for matmul).
 * If the weight is a GPUBuffer, returns it directly.
 * Otherwise, allocates a new buffer and uploads the data.
 *
 * @param {GPUBuffer | import('../../gpu/weight-buffer.js').WeightBuffer | import('../../gpu/weight-buffer.js').CpuWeightBuffer | Float32Array | ArrayBuffer} weight - Weight data (GPUBuffer, WeightBuffer, CpuWeightBuffer, or CPU array)
 * @param {string} label - Debug label for the buffer
 * @returns {GPUBuffer | import('../../gpu/weight-buffer.js').WeightBuffer}
 */
export function getWeightBuffer(weight, label) {
  // Preserve WeightBuffer to maintain dtype/layout for matmul
  if (isWeightBuffer(weight)) {
    return weight;
  }
  if (weight instanceof GPUBuffer) {
    return weight;
  }

  const device = getDevice();
  if (!device) {
    throw new Error('No GPU device available for weight buffer creation');
  }

  /** @type {Float32Array} */
  let data;
  if (isCpuWeightBuffer(weight)) {
    data = weight.data;
  } else if (weight instanceof Float32Array) {
    data = weight;
  } else {
    data = new Float32Array(/** @type {ArrayBuffer} */ (weight));
  }

  const buf = acquireBuffer(data.byteLength, undefined, label);
  device.queue.writeBuffer(buf, 0, /** @type {BufferSource} */ (/** @type {unknown} */ (data)));
  return buf;
}

/**
 * Get or create GPU buffer for RMSNorm weight tensor.
 *
 * RMSNorm applies the +1 offset at runtime when configured, so weights are
 * uploaded as-is without modifying values here.
 *
 * @param {GPUBuffer | Float32Array | ArrayBuffer | { buffer: ArrayBuffer; byteOffset: number; byteLength: number } | import('../../gpu/weight-buffer.js').CpuWeightBuffer} weight - Weight data (GPUBuffer or CPU array)
 * @param {string} label - Debug label for the buffer
 * @param {import('./weights.js').WeightBufferConfig} config - Weight buffer configuration
 * @param {import('./weights.js').WeightDebugFlags} [debugFlags] - Mutable debug flags (optional)
 * @returns {GPUBuffer}
 */
export function getNormWeightBuffer(weight, label, config, debugFlags) {
  // Debug: Log whether weight is GPUBuffer (first time only)
  if (debugFlags && !debugFlags.normBufferTypeLogged) {
    debugFlags.normBufferTypeLogged = true;
    log.debug('Weights', `getNormWeightBuffer: weight is GPUBuffer=${weight instanceof GPUBuffer}, label=${label}`);
  }

  if (weight instanceof GPUBuffer) {
    // If already a GPUBuffer, we can't modify it - assume it was preprocessed
    return weight;
  }

  const device = getDevice();
  if (!device) {
    throw new Error('No GPU device available for norm weight buffer creation');
  }

  // RMSNorm weight offset is handled in the kernel, so upload raw weights as-is.

  // Standard path: just copy to GPU
  /** @type {Float32Array} */
  let data;
  if (isCpuWeightBuffer(weight)) {
    data = weight.data;
  } else if (weight instanceof Float32Array) {
    data = weight;
  } else if ('buffer' in weight && 'byteOffset' in weight && 'byteLength' in weight) {
    data = new Float32Array(weight.buffer, weight.byteOffset, weight.byteLength / 4);
  } else {
    data = new Float32Array(/** @type {ArrayBuffer} */ (weight));
  }

  const buf = acquireBuffer(data.byteLength, undefined, label);
  device.queue.writeBuffer(buf, 0, /** @type {BufferSource} */ (/** @type {unknown} */ (data)));
  return buf;
}

/**
 * Get GPU weight buffer, ensuring it's on GPU.
 *
 * This is primarily used in batched command paths where we expect
 * weights to already be on GPU. If not, logs a warning and uploads.
 *
 * @param {GPUBuffer | import('../../gpu/weight-buffer.js').WeightBuffer | import('../../gpu/weight-buffer.js').CpuWeightBuffer | Float32Array | ArrayBuffer} weight - Weight data (should be GPUBuffer or WeightBuffer)
 * @param {string} label - Debug label for the buffer
 * @returns {GPUBuffer}
 */
export function getGPUWeightBuffer(weight, label) {
  // Handle WeightBuffer by extracting underlying GPUBuffer
  if (isWeightBuffer(weight)) {
    return weight.buffer;
  }
  if (weight instanceof GPUBuffer) {
    return weight;
  }
  // Weight not on GPU - this shouldn't happen if loader is working correctly
  log.warn('Weights', `Weight ${label} not on GPU, uploading`);
  // At this point weight is Float32Array or ArrayBuffer, so getWeightBuffer returns GPUBuffer
  return /** @type {GPUBuffer} */ (getWeightBuffer(weight, label));
}

// ============================================================================
// Weight Buffer Factory
// ============================================================================

/**
 * Create weight buffer helper functions bound to a specific config.
 *
 * This factory creates helper functions that can be passed to other
 * pipeline modules, avoiding the need to pass config everywhere.
 *
 * @param {import('./weights.js').WeightBufferConfig} config - Weight buffer configuration
 * @param {import('./weights.js').WeightDebugFlags} [debugFlags] - Mutable debug flags (optional)
 * @returns {{ getWeightBuffer: (weight: GPUBuffer | import('../../gpu/weight-buffer.js').WeightBuffer | import('../../gpu/weight-buffer.js').CpuWeightBuffer | Float32Array | ArrayBuffer, label: string) => GPUBuffer | import('../../gpu/weight-buffer.js').WeightBuffer, getNormWeightBuffer: (weight: GPUBuffer | Float32Array | ArrayBuffer | import('../../gpu/weight-buffer.js').CpuWeightBuffer, label: string) => GPUBuffer, getGPUWeightBuffer: (weight: GPUBuffer | import('../../gpu/weight-buffer.js').WeightBuffer | import('../../gpu/weight-buffer.js').CpuWeightBuffer | Float32Array | ArrayBuffer, label: string) => GPUBuffer }}
 */
export function createWeightBufferHelpers(config, debugFlags) {
  return {
    /**
     * Get or create GPU buffer for a weight tensor.
     */
    getWeightBuffer: (weight, label) =>
      getWeightBuffer(weight, label),

    /**
     * Get or create GPU buffer for RMSNorm weight tensor.
     */
    getNormWeightBuffer: (weight, label) =>
      getNormWeightBuffer(weight, label, config, debugFlags),

    /**
     * Get GPU weight buffer, ensuring it's on GPU.
     */
    getGPUWeightBuffer: (weight, label) =>
      getGPUWeightBuffer(weight, label),
  };
}

// ============================================================================
// Batch Buffer Tracking
// ============================================================================

/**
 * Buffer tracking for batched command execution.
 *
 * Tracks temporary buffers that need to be released after a batch is submitted.
 */
export class BatchBufferTracker {
  constructor() {
    /** @type {GPUBuffer[]} */
    this._buffersToRelease = [];
  }

  /**
   * Track a temporary buffer for cleanup after batch submit.
   *
   * @param {GPUBuffer | Float32Array | ArrayBuffer} buffer - Buffer to track (only GPUBuffers are tracked)
   */
  track(buffer) {
    if (buffer instanceof GPUBuffer) {
      this._buffersToRelease.push(buffer);
    }
  }

  /**
   * Get all tracked buffers.
   *
   * @returns {GPUBuffer[]}
   */
  getTracked() {
    return this._buffersToRelease;
  }

  /**
   * Clear tracked buffers (call after releasing them).
   */
  clear() {
    this._buffersToRelease = [];
  }
}
