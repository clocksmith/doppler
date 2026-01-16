/**
 * Weight Downcast - F32 to F16 weight conversion utility.
 *
 * Provides a unified utility for downcasting F32 weights to F16 when
 * the GPU supports shader-f16. Used by layer loading, expert loading,
 * embedding loading, and final weights loading.
 *
 * @module loader/weight-downcast
 */

import type { WeightBuffer, WeightLayout } from '../gpu/weight-buffer.js';

export interface DowncastOptions {
  /** Label for debugging (e.g., "qProj", "lmHead") */
  label: string;
  /** Keep F32 weights (skip downcast) */
  keepF32: boolean;
  /** Resolved dtype for non-WeightBuffer inputs */
  dtype?: string;
  /** Shape for the resulting WeightBuffer */
  shape?: number[];
  /** Layout preference (defaults to preserving existing or 'row') */
  layout?: WeightLayout;
  /** Layer index for debug logging */
  layerIdx?: number;
}

export interface DowncastResult {
  /** The resulting buffer (may be original if no downcast) */
  buffer: GPUBuffer | WeightBuffer;
  /** Whether downcast was performed */
  wasDowncast: boolean;
  /** New GPU buffer allocated (caller should track for cleanup) */
  newBuffer: GPUBuffer | null;
}

/**
 * Attempt to downcast a weight buffer from F32 to F16.
 *
 * @param buf - Input buffer (GPUBuffer or WeightBuffer)
 * @param options - Downcast options
 * @returns Downcast result, or null if input is null/unsupported
 */
export declare function maybeDowncastToF16(
  buf: GPUBuffer | WeightBuffer | null,
  options: DowncastOptions
): Promise<DowncastResult | null>;

/**
 * Downcast multiple weight buffers, tracking new GPU buffers.
 *
 * @param weights - Record of weight buffers to downcast
 * @param keys - Keys to downcast
 * @param options - Base options (label will be set per key)
 * @param gpuBuffers - Set to track new GPU buffers
 */
export declare function batchDowncastWeights<T extends Record<string, GPUBuffer | WeightBuffer | null>>(
  weights: T,
  keys: (keyof T)[],
  options: Omit<DowncastOptions, 'label'>,
  gpuBuffers: Set<GPUBuffer>
): Promise<void>;
