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
  type WeightBuffer,
  type WeightLayout,
} from '../gpu/weight-buffer.js';
import { trace as debugTrace } from '../debug/index.js';

// ============================================================================
// Types
// ============================================================================

export interface DowncastOptions {
  /** Label for debugging (e.g., "qProj", "lmHead") */
  label: string;
  /** Keep F32 weights (skip downcast) */
  keepF32: boolean;
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

// ============================================================================
// Main Downcast Function
// ============================================================================

/**
 * Attempt to downcast a weight buffer from F32 to F16.
 *
 * @param buf - Input buffer (GPUBuffer or WeightBuffer)
 * @param options - Downcast options
 * @returns Downcast result, or null if input is null/unsupported
 */
export async function maybeDowncastToF16(
  buf: GPUBuffer | WeightBuffer | null,
  options: DowncastOptions
): Promise<DowncastResult | null> {
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
 */
async function downcastWeightBuffer(
  buf: WeightBuffer,
  options: DowncastOptions
): Promise<DowncastResult> {
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
    const shape = options.shape ?? (buf.shape as number[]);
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
    debugTrace.loader(`Failed to downcast ${options.label} to f16: ${(e as Error).message}`);
    return {
      buffer: buf,
      wasDowncast: false,
      newBuffer: null,
    };
  }
}

/**
 * Downcast a raw GPUBuffer from F32 to F16.
 */
async function downcastGPUBuffer(
  buf: GPUBuffer,
  options: DowncastOptions
): Promise<DowncastResult> {
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
    debugTrace.loader(`Failed to downcast ${options.label} to f16: ${(e as Error).message}`);
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
 * @param weights - Record of weight buffers to downcast
 * @param keys - Keys to downcast
 * @param options - Base options (label will be set per key)
 * @param gpuBuffers - Set to track new GPU buffers
 */
export async function batchDowncastWeights<T extends Record<string, GPUBuffer | WeightBuffer | null>>(
  weights: T,
  keys: (keyof T)[],
  options: Omit<DowncastOptions, 'label'>,
  gpuBuffers: Set<GPUBuffer>
): Promise<void> {
  for (const key of keys) {
    const buf = weights[key];
    if (!buf) continue;

    const result = await maybeDowncastToF16(buf as GPUBuffer | WeightBuffer, {
      ...options,
      label: String(key),
    });

    if (result?.wasDowncast) {
      (weights as Record<string, unknown>)[key as string] = result.buffer;
      if (result.newBuffer) {
        gpuBuffers.add(result.newBuffer);
      }
    }
  }
}
