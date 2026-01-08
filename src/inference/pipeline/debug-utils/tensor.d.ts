/**
 * Tensor inspection utilities for kernel step debugging.
 *
 * Provides GPU buffer readback and statistics computation for debugging.
 * These operations are expensive (require GPU sync) - use sparingly.
 *
 * Enable with setDebugCategories({ kernel: true }, { bufferStats: true })
 * or use DEBUG_PRESETS.kernelStep
 *
 * @example
 * // Enable kernel step debugging for layer 0 only
 * setDebugCategories({ kernel: true }, { layers: [0], bufferStats: true });
 *
 * // In pipeline code:
 * if (isKernelDebugEnabled(layerIdx)) {
 *   await dumpTensor(outputBuffer, 'matmul_output', { layerIdx });
 * }
 *
 * @module inference/pipeline/debug-utils/tensor
 */

import type { KVCache, SlidingWindowKVCache } from '../../kv-cache.js';

// ============================================================================
// Types
// ============================================================================

/** Tensor statistics from GPU readback */
export interface TensorStats {
  shape: string;
  dtype: string;
  min: number;
  max: number;
  maxAbs: number;
  mean: number;
  nonZero: number;
  total: number;
  nanCount: number;
  infCount: number;
  sample: number[];
}

// ============================================================================
// Tensor Inspection Functions
// ============================================================================

/**
 * Dump a GPU tensor's contents for debugging.
 * This is expensive (requires GPU sync + readback) - use sparingly.
 *
 * @param buffer - GPU buffer to inspect
 * @param label - Descriptive label for logging
 * @param options - Additional options
 */
export function dumpTensor(
  buffer: GPUBuffer,
  label: string,
  options?: {
    layerIdx?: number;
    shape?: [number, number] | [number];
    dtype?: 'f32' | 'f16' | 'bf16';
    sampleCount?: number;
    warnThreshold?: number;
  }
): Promise<TensorStats | null>;

/**
 * Dump stats for a single token row within a 2D [numTokens, rowSize] buffer.
 * Use this when matching per-token reference implementations (e.g., HuggingFace hooks).
 */
export function dumpTokenVector(
  buffer: GPUBuffer,
  label: string,
  options: {
    layerIdx?: number;
    tokenIdx: number;
    rowSize: number;
    dtype?: 'f32' | 'f16' | 'bf16';
    sampleCount?: number;
    warnThreshold?: number;
  }
): Promise<TensorStats | null>;

/**
 * Log a kernel step with optional tensor dump.
 * Use this after kernel invocations to trace execution.
 *
 * @param kernelName - Name of the kernel (e.g., 'matmul', 'rmsnorm')
 * @param info - Additional info to log
 */
export function logKernelStep(
  kernelName: string,
  info: {
    layerIdx?: number;
    M?: number;
    N?: number;
    K?: number;
    size?: number;
    label?: string;
  }
): void;

/**
 * Dump KV cache state for a specific layer.
 * Reads both keys and values buffers and reports statistics.
 *
 * @param kvCache - KV cache instance
 * @param layerIdx - Layer index to inspect
 */
export function dumpKVCache(
  kvCache: KVCache | SlidingWindowKVCache,
  layerIdx: number
): Promise<{ keys: TensorStats | null; values: TensorStats | null } | null>;

/**
 * Check if kernel step debugging is enabled.
 * Use this to gate expensive debug operations.
 */
export function isKernelDebugEnabled(layerIdx?: number): boolean;
