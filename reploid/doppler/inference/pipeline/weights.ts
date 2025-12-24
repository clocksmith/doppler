/**
 * Weight buffer management utilities.
 *
 * This module handles:
 * - Creating GPU buffers from CPU weight data
 * - Applying weight transformations (e.g., Gemma norm offset)
 * - Type guards for layer weight structures
 * - Buffer lifecycle management
 *
 * @module inference/pipeline/weights
 */

import { getDevice } from '../../gpu/device.js';
import { acquireBuffer } from '../../gpu/buffer-pool.js';
import type { LayerWeights, MaybeGPUBuffer } from './types.js';

// ============================================================================
// Types
// ============================================================================

/**
 * Configuration for weight buffer operations.
 */
export interface WeightBufferConfig {
  /** Whether to apply +1 offset for Gemma-style norm weights */
  rmsNormWeightOffset: boolean;
}

/**
 * Debug flags for weight buffer operations.
 */
export interface WeightDebugFlags {
  normBufferTypeLogged?: boolean;
  normOffsetDebugDone?: boolean;
}

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
 * @param value - Value to check
 * @returns True if value is a LayerWeights object
 */
export function isLayerWeights(value: unknown): value is LayerWeights {
  return value !== null && typeof value === 'object' && !ArrayBuffer.isView(value) && !('getMappedRange' in (value as object));
}

/**
 * Get layer weights from weights map with type narrowing.
 *
 * @param weights - Map of weight names to weight data
 * @param key - Key to look up (e.g., "layer_0")
 * @returns LayerWeights if found and valid, null otherwise
 */
export function getLayerWeights(
  weights: Map<string, LayerWeights | Float32Array | GPUBuffer>,
  key: string
): LayerWeights | null {
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
 * If the weight is already a GPUBuffer, returns it directly.
 * Otherwise, allocates a new buffer and uploads the data.
 *
 * @param weight - Weight data (GPUBuffer or CPU array)
 * @param label - Debug label for the buffer
 * @returns GPUBuffer ready for use
 */
export function getWeightBuffer(
  weight: GPUBuffer | Float32Array | ArrayBuffer,
  label: string
): GPUBuffer {
  if (weight instanceof GPUBuffer) {
    return weight;
  }
  const device = getDevice();
  if (!device) {
    throw new Error('No GPU device available for weight buffer creation');
  }

  const data = weight instanceof Float32Array ? weight : new Float32Array(weight);
  const buf = acquireBuffer(data.byteLength, undefined, label);
  device.queue.writeBuffer(buf, 0, data as unknown as BufferSource);
  return buf;
}

/**
 * Get or create GPU buffer for RMSNorm weight tensor.
 *
 * Applies the +1 offset for Gemma 3+ models which use (1 + weight) in
 * the norm formula. This transformation is only applied if the config
 * specifies rmsNormWeightOffset=true.
 *
 * @param weight - Weight data (GPUBuffer or CPU array)
 * @param label - Debug label for the buffer
 * @param config - Weight buffer configuration
 * @param debugFlags - Mutable debug flags (optional)
 * @returns GPUBuffer ready for use
 */
export function getNormWeightBuffer(
  weight: GPUBuffer | Float32Array | ArrayBuffer | { buffer: ArrayBuffer; byteOffset: number; byteLength: number },
  label: string,
  config: WeightBufferConfig,
  debugFlags?: WeightDebugFlags
): GPUBuffer {
  // Debug: Log whether weight is GPUBuffer (first time only)
  if (debugFlags && !debugFlags.normBufferTypeLogged) {
    debugFlags.normBufferTypeLogged = true;
    console.log(`[DEBUG] getNormWeightBuffer: weight is GPUBuffer=${weight instanceof GPUBuffer}, label=${label}`);
  }

  if (weight instanceof GPUBuffer) {
    // If already a GPUBuffer, we can't modify it - assume it was preprocessed
    return weight;
  }

  const device = getDevice();
  if (!device) {
    throw new Error('No GPU device available for norm weight buffer creation');
  }

  // For Gemma 3+, apply the +1 offset: weight_effective = 1 + weight
  if (config.rmsNormWeightOffset) {
    // Debug: first time only
    if (debugFlags && !debugFlags.normOffsetDebugDone) {
      debugFlags.normOffsetDebugDone = true;
      console.log(`[DEBUG] WARNING: Applying +1 offset to norm weights in pipeline (may be duplicate!)`);
    }

    let f32Data: Float32Array;
    if (weight instanceof Float32Array) {
      f32Data = new Float32Array(weight.length);
      for (let i = 0; i < weight.length; i++) {
        f32Data[i] = 1.0 + weight[i];
      }
    } else if ('buffer' in weight && weight.buffer instanceof ArrayBuffer) {
      // Typed array view
      const src = new Float32Array(weight.buffer, weight.byteOffset, weight.byteLength / 4);
      f32Data = new Float32Array(src.length);
      for (let i = 0; i < src.length; i++) {
        f32Data[i] = 1.0 + src[i];
      }
    } else {
      // ArrayBuffer - interpret as F32
      const src = new Float32Array(weight as ArrayBuffer);
      f32Data = new Float32Array(src.length);
      for (let i = 0; i < src.length; i++) {
        f32Data[i] = 1.0 + src[i];
      }
    }

    const buf = acquireBuffer(f32Data.byteLength, undefined, label);
    device.queue.writeBuffer(buf, 0, f32Data as unknown as BufferSource);
    return buf;
  }

  // Standard path: just copy to GPU
  const data = weight instanceof Float32Array ? weight : new Float32Array(weight as ArrayBuffer);
  const buf = acquireBuffer(data.byteLength, undefined, label);
  device.queue.writeBuffer(buf, 0, data as unknown as BufferSource);
  return buf;
}

/**
 * Get GPU weight buffer, ensuring it's on GPU.
 *
 * This is primarily used in batched command paths where we expect
 * weights to already be on GPU. If not, logs a warning and uploads.
 *
 * @param weight - Weight data (should be GPUBuffer)
 * @param label - Debug label for the buffer
 * @returns GPUBuffer ready for use
 */
export function getGPUWeightBuffer(
  weight: GPUBuffer | Float32Array | ArrayBuffer,
  label: string
): GPUBuffer {
  if (weight instanceof GPUBuffer) {
    return weight;
  }
  // Weight not on GPU - this shouldn't happen if loader is working correctly
  console.warn(`[Pipeline] Weight ${label} not on GPU, uploading`);
  return getWeightBuffer(weight, label);
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
 * @param config - Weight buffer configuration
 * @param debugFlags - Mutable debug flags (optional)
 * @returns Object with bound helper functions
 */
export function createWeightBufferHelpers(
  config: WeightBufferConfig,
  debugFlags?: WeightDebugFlags
) {
  return {
    /**
     * Get or create GPU buffer for a weight tensor.
     */
    getWeightBuffer: (weight: GPUBuffer | Float32Array | ArrayBuffer, label: string) =>
      getWeightBuffer(weight, label),

    /**
     * Get or create GPU buffer for RMSNorm weight tensor.
     */
    getNormWeightBuffer: (weight: GPUBuffer | Float32Array | ArrayBuffer, label: string) =>
      getNormWeightBuffer(weight, label, config, debugFlags),

    /**
     * Get GPU weight buffer, ensuring it's on GPU.
     */
    getGPUWeightBuffer: (weight: GPUBuffer | Float32Array | ArrayBuffer, label: string) =>
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
  private buffersToRelease: GPUBuffer[] = [];

  /**
   * Track a temporary buffer for cleanup after batch submit.
   *
   * @param buffer - Buffer to track (only GPUBuffers are tracked)
   */
  track(buffer: GPUBuffer | Float32Array | ArrayBuffer): void {
    if (buffer instanceof GPUBuffer) {
      this.buffersToRelease.push(buffer);
    }
  }

  /**
   * Get all tracked buffers.
   *
   * @returns Array of tracked GPUBuffers
   */
  getTracked(): GPUBuffer[] {
    return this.buffersToRelease;
  }

  /**
   * Clear tracked buffers (call after releasing them).
   */
  clear(): void {
    this.buffersToRelease = [];
  }
}
