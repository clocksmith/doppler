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

import { getDevice, getKernelCapabilities } from '../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../gpu/buffer-pool.js';
import { dequantize, dequantizeQ6K, castF16ToF32, runBF16ToF16 } from '../gpu/kernel-selector.js';
import { createTensor } from '../gpu/tensor.js';
import {
  createWeightBuffer,
  type WeightBuffer,
  type WeightDtype,
  type WeightLayout,
} from '../gpu/weight-buffer.js';
import { f16ToF32, convertBF16ToF32GPU, shouldDequantizeToF16, applyBufferLayout } from './dtype-utils.js';
import { QK_K, Q4K_BLOCK_BYTES, Q6K_BLOCK_BYTES } from './quantization-constants.js';
import type { TensorLocation, KernelCapabilities } from './loader-types.js';
import { isTraceEnabled, trace as debugTrace } from '../debug/index.js';

// ============================================================================
// Types
// ============================================================================

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

// ============================================================================
// Q4K Detection
// ============================================================================

/**
 * Check if a Q4K tensor is packed (incompatible with fused matmul).
 */
export function isPackedQ4K(location: TensorLocation): boolean {
  if (!Array.isArray(location.shape) || location.shape.length !== 2) {
    return false;
  }
  const [rows, cols] = location.shape;
  const expectedRowwise = rows * Math.ceil(cols / QK_K) * Q4K_BLOCK_BYTES;
  return location.size < expectedRowwise;
}

/**
 * Check if tensor name indicates an embedding (excluded from fused Q4K).
 */
export function isEmbeddingName(name: string): boolean {
  const lower = name.toLowerCase();
  return lower.includes('embd') || lower.includes('embed') || lower.includes('wte');
}

/**
 * Determine if fused Q4K path should be used for a tensor.
 */
export function shouldUseFusedQ4K(
  name: string,
  location: TensorLocation,
  config: TensorLoadConfig
): boolean {
  if (!config.useFusedQ4K) return false;

  const caps = config.gpuCapabilities || getKernelCapabilities();
  if (!caps?.hasSubgroups) return false;

  const isMatmulWeight = shouldDequantizeToF16(name);
  if (!isMatmulWeight) return false;

  if (isEmbeddingName(name)) return false;
  if (isPackedQ4K(location)) return false;

  return true;
}

// ============================================================================
// Dtype Output Selection
// ============================================================================

/**
 * Determine output dtype for dequantized Q4K tensor.
 */
export function getQ4KOutputDtype(
  name: string,
  config: TensorLoadConfig
): 'f16' | 'f32' {
  const isMatmulWeight = shouldDequantizeToF16(name);
  if (!isMatmulWeight) return 'f32';

  if (config.keepF32Weights) return 'f32';

  const caps = config.gpuCapabilities || getKernelCapabilities();
  return caps?.hasF16 ? 'f16' : 'f32';
}

/**
 * Determine weight layout based on config and tensor type.
 */
export function getWeightLayout(
  name: string,
  location: TensorLocation,
  config: TensorLoadConfig
): WeightLayout {
  if (location.layout === 'column') return 'column';

  const isMatmulWeight = shouldDequantizeToF16(name);
  if (config.q4kLayout === 'column_wise' && isMatmulWeight) {
    return 'column';
  }

  return 'row';
}

// ============================================================================
// CPU Path Helpers
// ============================================================================

/**
 * Convert BF16 data to F32 on CPU.
 */
export function convertBF16ToF32CPU(bf16Data: Uint16Array): Float32Array {
  const f32 = new Float32Array(bf16Data.length);
  const tmp = new ArrayBuffer(4);
  const u32View = new Uint32Array(tmp);
  const f32View = new Float32Array(tmp);

  for (let i = 0; i < bf16Data.length; i++) {
    u32View[0] = bf16Data[i] << 16;
    f32[i] = f32View[0];
  }

  return f32;
}

/**
 * Convert F16 data to F32 on CPU.
 */
export function convertF16ToF32CPU(f16Data: Uint16Array): Float32Array {
  const f32 = new Float32Array(f16Data.length);
  for (let i = 0; i < f16Data.length; i++) {
    f32[i] = f16ToF32(f16Data[i]);
  }
  return f32;
}

// ============================================================================
// GPU Tensor Loading
// ============================================================================

/**
 * Load Q4K tensor to GPU with fused path (keeps raw quantized data).
 */
export async function loadQ4KFused(
  shardData: Uint8Array,
  location: TensorLocation,
  name: string
): Promise<TensorLoadResult> {
  const device = getDevice()!;
  const buffer = acquireBuffer(location.size, undefined, `q4k_${name}`);
  device.queue.writeBuffer(buffer, 0, shardData as unknown as BufferSource);

  return {
    data: createWeightBuffer(buffer, 'q4k', 'row', location.shape, name),
    allocatedBuffers: [buffer],
  };
}

/**
 * Load Q4K tensor to GPU with dequantization.
 */
export async function loadQ4KDequant(
  shardData: Uint8Array,
  location: TensorLocation,
  name: string,
  config: TensorLoadConfig
): Promise<TensorLoadResult> {
  const device = getDevice()!;
  const quantBuffer = acquireBuffer(location.size, undefined, `quant_${name}`);
  device.queue.writeBuffer(quantBuffer, 0, shardData as unknown as BufferSource);

  const numBlocks = Math.ceil(location.size / 144);
  const outputDtype = getQ4KOutputDtype(name, config);

  debugTrace.loader(
    `Dequantizing ${name}: size=${location.size}, numBlocks=${numBlocks}, ` +
    `outputDtype=${outputDtype}, expectedOutput=${numBlocks * 256 * (outputDtype === 'f16' ? 2 : 4)}`
  );

  const dequantizedTensor = await dequantize(quantBuffer, numBlocks, { outputDtype });
  const dequantized = dequantizedTensor.buffer;

  debugTrace.loader(`Dequantized ${name}: resultSize=${dequantized.size}`);
  releaseBuffer(quantBuffer);

  const layout = getWeightLayout(name, location, config);
  const dtype: WeightDtype = outputDtype;

  return {
    data: createWeightBuffer(dequantized, dtype, layout, location.shape, name),
    allocatedBuffers: [dequantized],
  };
}

/**
 * Load Q6K tensor to GPU.
 */
export async function loadQ6K(
  shardData: Uint8Array,
  location: TensorLocation,
  name: string
): Promise<TensorLoadResult> {
  const device = getDevice()!;

  debugTrace.loader(`Loading Q6_K tensor "${name}", size=${location.size}`);
  const quantBuffer = acquireBuffer(location.size, undefined, `quant_${name}`);
  device.queue.writeBuffer(quantBuffer, 0, shardData as unknown as BufferSource);

  const numBlocks = Math.floor(location.size / Q6K_BLOCK_BYTES);
  debugTrace.loader(
    `Dequantizing Q6_K ${name}: size=${location.size}, numBlocks=${numBlocks}, ` +
    `expectedOutput=${numBlocks * 256 * 2} (f16)`
  );

  const dequantizedTensor = await dequantizeQ6K(quantBuffer, numBlocks, { outputDtype: 'f16' });
  const dequantized = dequantizedTensor.buffer;

  debugTrace.loader(`Dequantized Q6_K ${name}: resultSize=${dequantized.size}`);
  releaseBuffer(quantBuffer);

  const isMatmulWeight = shouldDequantizeToF16(name);
  if (isMatmulWeight) {
    return {
      data: createWeightBuffer(dequantized, 'f16', 'row', location.shape, name),
      allocatedBuffers: [dequantized],
    };
  }

  return {
    data: dequantized,
    allocatedBuffers: [dequantized],
  };
}

/**
 * Load BF16 tensor to GPU.
 */
export async function loadBF16(
  shardData: Uint8Array,
  location: TensorLocation,
  name: string,
  config: TensorLoadConfig
): Promise<TensorLoadResult> {
  const device = getDevice()!;
  const srcBuffer = acquireBuffer(shardData.byteLength, undefined, `${name}_bf16`);
  device.queue.writeBuffer(srcBuffer, 0, shardData as unknown as BufferSource);

  const numElements = shardData.byteLength / 2;
  const caps = config.gpuCapabilities || getKernelCapabilities();
  const isMatmulWeight = shouldDequantizeToF16(name);

  // For matmul weights with F16 support: BF16 → F16 directly
  if (caps?.hasF16 && isMatmulWeight) {
    const f16Tensor = await runBF16ToF16(srcBuffer, [numElements], name);
    releaseBuffer(srcBuffer);
    debugTrace.loader(`BF16→F16 for matmul weight: ${name} (${numElements} elements)`);

    const layout: WeightLayout = location.layout === 'column' ? 'column' : 'row';
    return {
      data: createWeightBuffer(f16Tensor.buffer, 'f16', layout, location.shape, name),
      allocatedBuffers: [f16Tensor.buffer],
    };
  }

  // Standard path: BF16 → F32
  const dstBuffer = await convertBF16ToF32GPU(srcBuffer, numElements, name);
  releaseBuffer(srcBuffer);

  if (dstBuffer instanceof GPUBuffer) {
    if (isMatmulWeight) {
      const layout: WeightLayout = location.layout === 'column' ? 'column' : 'row';
      return {
        data: createWeightBuffer(dstBuffer, 'f32', layout, location.shape, name),
        allocatedBuffers: [dstBuffer],
      };
    }
    return {
      data: applyBufferLayout(dstBuffer, location),
      allocatedBuffers: [dstBuffer],
    };
  }

  // Float32Array returned (shouldn't happen in GPU path)
  return {
    data: dstBuffer,
    allocatedBuffers: [],
  };
}

/**
 * Load F16/F32 tensor to GPU.
 */
export async function loadFloat(
  shardData: Uint8Array,
  location: TensorLocation,
  name: string
): Promise<TensorLoadResult> {
  const device = getDevice()!;
  const buffer = acquireBuffer(location.size, undefined, name);
  device.queue.writeBuffer(buffer, 0, shardData as unknown as BufferSource);

  const dtype: WeightDtype = location.dtype === 'F16' ? 'f16' : 'f32';
  const layout: WeightLayout = location.layout === 'column' ? 'column' : 'row';
  const isMatmulWeight = shouldDequantizeToF16(name);

  // Return WeightBuffer for matmul weights
  if (isMatmulWeight) {
    return {
      data: createWeightBuffer(buffer, dtype, layout, location.shape, name),
      allocatedBuffers: [buffer],
    };
  }

  // Non-matmul F16 weights need upcast to F32
  if (dtype === 'f16') {
    const numElements = location.shape.reduce((a, b) => a * b, 1);
    const inputTensor = createTensor(buffer, 'f16', [numElements], `${name}_f16`);
    const f32Tensor = await castF16ToF32(inputTensor);
    releaseBuffer(buffer);
    return {
      data: applyBufferLayout(f32Tensor.buffer, location),
      allocatedBuffers: [f32Tensor.buffer],
    };
  }

  return {
    data: applyBufferLayout(buffer, location),
    allocatedBuffers: [buffer],
  };
}

// ============================================================================
// Main GPU Loading Entry Point
// ============================================================================

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
export async function loadTensorToGPU(
  shardData: Uint8Array,
  location: TensorLocation,
  name: string,
  config: TensorLoadConfig
): Promise<TensorLoadResult> {
  const dtype = location.dtype;

  // Q4_K / Q4_K_M
  if (dtype === 'Q4_K_M' || dtype === 'Q4_K') {
    if (shouldUseFusedQ4K(name, location, config)) {
      debugTrace.loader(`Loading Q4K weight (fused): ${name} (size=${location.size})`);
      return loadQ4KFused(shardData, location, name);
    }

    if (config.useFusedQ4K && isPackedQ4K(location)) {
      const [rows, cols] = location.shape;
      debugTrace.loader(`Packed Q4K weight ${name} [${rows},${cols}] incompatible with fused matmul, using dequant`);
    }

    return loadQ4KDequant(shardData, location, name, config);
  }

  // Q6_K
  if (dtype === 'Q6_K') {
    return loadQ6K(shardData, location, name);
  }

  // BF16
  if (dtype === 'BF16') {
    return loadBF16(shardData, location, name, config);
  }

  // F16 / F32
  return loadFloat(shardData, location, name);
}

/**
 * Load tensor data on CPU (no GPU upload).
 *
 * @param shardData - Raw tensor data from shard(s)
 * @param location - Tensor location info
 * @returns CPU tensor data
 */
export function loadTensorToCPU(
  shardData: Uint8Array,
  location: TensorLocation
): Float32Array | Uint8Array {
  const dtype = location.dtype;

  // Quantized data - return raw for CPU
  if (dtype === 'Q4_K_M' || dtype === 'Q4_K' || dtype === 'Q6_K') {
    return shardData;
  }

  // BF16 - convert to F32
  if (dtype === 'BF16') {
    const bf16 = new Uint16Array(shardData.slice().buffer);
    return convertBF16ToF32CPU(bf16);
  }

  // F16 - convert to F32
  if (dtype === 'F16') {
    const f16 = new Uint16Array(shardData.slice().buffer);
    return convertF16ToF32CPU(f16);
  }

  // F32 - return as Float32Array
  return new Float32Array(shardData.slice().buffer);
}
