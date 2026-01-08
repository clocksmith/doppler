/**
 * Tensor Loader - Dtype-specific tensor loading and conversion.
 *
 * Handles loading tensors from shards with support for:
 * - Q4_K/Q4_K_M quantized tensors (fused and dequant paths)
 * - Q6_K quantized tensors
 * - BF16 tensors (GPU and CPU conversion)
 * - F16/F32 tensors
 *
 * @module loader/tensor-loader
 */

import type { WeightBuffer, WeightLayout } from '../gpu/weight-buffer.js';
import type { TensorLocation, KernelCapabilities } from './loader-types.js';

export interface TensorLoadConfig {
  /** Use fused Q4K matmul kernels */
  useFusedQ4K: boolean;
  /** Keep weights as F32 (disable F16 downcasting) */
  keepF32Weights: boolean;
  /** Q4K layout from manifest */
  q4kLayout: 'flat' | 'row_wise' | 'column_wise' | null;
  /** GPU capabilities */
  gpuCapabilities: KernelCapabilities | null;
}

export interface TensorLoadResult {
  /** The loaded tensor data */
  data: GPUBuffer | WeightBuffer | Float32Array | Uint8Array;
  /** GPU buffers that were allocated (caller should track for cleanup) */
  allocatedBuffers: GPUBuffer[];
}

/**
 * Check if a Q4K tensor is packed (incompatible with fused matmul).
 */
export declare function isPackedQ4K(location: TensorLocation): boolean;

/**
 * Check if tensor name indicates an embedding (excluded from fused Q4K).
 */
export declare function isEmbeddingName(name: string): boolean;

/**
 * Determine if fused Q4K path should be used for a tensor.
 */
export declare function shouldUseFusedQ4K(
  name: string,
  location: TensorLocation,
  config: TensorLoadConfig
): boolean;

/**
 * Determine output dtype for dequantized Q4K tensor.
 */
export declare function getQ4KOutputDtype(
  name: string,
  config: TensorLoadConfig
): 'f16' | 'f32';

/**
 * Determine weight layout based on config and tensor type.
 */
export declare function getWeightLayout(
  name: string,
  location: TensorLocation,
  config: TensorLoadConfig
): WeightLayout;

/**
 * Convert BF16 data to F32 on CPU.
 */
export declare function convertBF16ToF32CPU(bf16Data: Uint16Array): Float32Array;

/**
 * Convert F16 data to F32 on CPU.
 */
export declare function convertF16ToF32CPU(f16Data: Uint16Array): Float32Array;

/**
 * Load Q4K tensor to GPU with fused path (keeps raw quantized data).
 */
export declare function loadQ4KFused(
  shardData: Uint8Array,
  location: TensorLocation,
  name: string
): Promise<TensorLoadResult>;

/**
 * Load Q4K tensor to GPU with dequantization.
 */
export declare function loadQ4KDequant(
  shardData: Uint8Array,
  location: TensorLocation,
  name: string,
  config: TensorLoadConfig
): Promise<TensorLoadResult>;

/**
 * Load Q6K tensor to GPU.
 */
export declare function loadQ6K(
  shardData: Uint8Array,
  location: TensorLocation,
  name: string
): Promise<TensorLoadResult>;

/**
 * Load BF16 tensor to GPU.
 */
export declare function loadBF16(
  shardData: Uint8Array,
  location: TensorLocation,
  name: string,
  config: TensorLoadConfig
): Promise<TensorLoadResult>;

/**
 * Load F16/F32 tensor to GPU.
 */
export declare function loadFloat(
  shardData: Uint8Array,
  location: TensorLocation,
  name: string
): Promise<TensorLoadResult>;

/**
 * Load tensor data to GPU based on dtype.
 *
 * Routes to appropriate handler based on tensor dtype.
 *
 * @param shardData - Raw tensor data from shard(s)
 * @param location - Tensor location info
 * @param name - Tensor name
 * @param config - Load configuration
 * @returns Loaded tensor result with allocated buffers
 */
export declare function loadTensorToGPU(
  shardData: Uint8Array,
  location: TensorLocation,
  name: string,
  config: TensorLoadConfig
): Promise<TensorLoadResult>;

/**
 * Load tensor data on CPU (no GPU upload).
 *
 * @param shardData - Raw tensor data from shard(s)
 * @param location - Tensor location info
 * @returns CPU tensor data
 */
export declare function loadTensorToCPU(
  shardData: Uint8Array,
  location: TensorLocation
): Float32Array | Uint8Array;
