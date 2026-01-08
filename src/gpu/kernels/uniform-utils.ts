/**
 * Uniform Utils - Uniform buffer creation utilities
 *
 * Provides utilities for creating and caching uniform buffers
 * for kernel dispatch.
 *
 * @module gpu/kernels/uniform-utils
 */

import { getDevice } from '../device.js';
import { getUniformCache } from '../uniform-cache.js';
import type { CommandRecorder } from '../command-recorder.js';

// ============================================================================
// Types
// ============================================================================

/** Options for uniform buffer creation */
export interface UniformBufferOptions {
  /** Use content-addressed cache for reuse (default: true) */
  useCache?: boolean;
}

// ============================================================================
// Uniform Buffer Creation
// ============================================================================

/**
 * Create a uniform buffer from raw data.
 * Uses caching by default for content-addressed reuse.
 */
export function createUniformBufferFromData(
  label: string,
  data: ArrayBuffer | ArrayBufferView,
  recorder?: CommandRecorder | null,
  deviceOverride?: GPUDevice | null,
  options?: UniformBufferOptions
): GPUBuffer {
  if (recorder) {
    return recorder.createUniformBuffer(data, label);
  }

  // Convert ArrayBufferView to ArrayBuffer for caching
  const arrayBuffer = data instanceof ArrayBuffer
    ? data
    : data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);

  // Use cache by default for non-recorder paths
  const useCache = options?.useCache ?? true;
  if (useCache && !deviceOverride) {
    return getUniformCache().getOrCreate(arrayBuffer, label);
  }

  // Fallback to direct creation (for custom device or explicit no-cache)
  const device = deviceOverride ?? getDevice();
  if (!device) {
    throw new Error('GPU device not initialized');
  }

  const byteLength = arrayBuffer.byteLength;
  const buffer = device.createBuffer({
    label,
    size: byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, arrayBuffer);
  return buffer;
}

/**
 * Create a uniform buffer with a DataView writer callback.
 * Allows structured data writing with proper alignment.
 */
export function createUniformBufferWithView(
  label: string,
  byteLength: number,
  writer: (view: DataView) => void,
  recorder?: CommandRecorder | null,
  deviceOverride?: GPUDevice | null
): GPUBuffer {
  const data = new ArrayBuffer(byteLength);
  const view = new DataView(data);
  writer(view);
  return createUniformBufferFromData(label, data, recorder, deviceOverride);
}
