/**
 * Dequantization Kernels
 *
 * Provides dequantization operations for:
 * - Q4_K_M quantization (GGUF format)
 * - MXFP4 quantization (GPT-OSS format)
 * - F16/F32 output support
 * - Subgroup and shared memory variants
 */

import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions, OutputDtypeOptions, OutputOffsetOptions, Vec4Options } from './types.js';

/** Dequantization kernel options */
export interface DequantOptions extends OutputBufferOptions, OutputOffsetOptions, OutputDtypeOptions, Vec4Options {
  groupSize?: number;
}

/**
 * Select the best dequantization kernel variant
 */
export declare function selectDequantKernel(options?: DequantOptions): string;

/**
 * Create bind group layout for dequant operation
 */
export declare function createDequantBindGroupLayout(): GPUBindGroupLayout;

/**
 * Run Q4_K_M dequantization
 */
export declare function dequantize(
  quantized: GPUBuffer,
  numBlocks: number,
  options?: DequantOptions
): Promise<Tensor>;

/**
 * Dequantize MXFP4 weights (GPT-OSS format)
 */
export declare function dequantizeMXFP4(
  blocks: GPUBuffer,
  scales: GPUBuffer,
  totalElements: number,
  numGroups: number,
  options?: DequantOptions
): Promise<Tensor>;

/**
 * Dequantize MXFP4 expert weights (extracts single expert from packed tensor)
 */
export declare function dequantizeMXFP4Expert(
  blocks: GPUBuffer,
  scales: GPUBuffer,
  expertIdx: number,
  numExperts: number,
  outDim: number,
  numGroups: number,
  options?: DequantOptions
): Promise<Tensor>;

/**
 * Run Q6_K dequantization
 */
export declare function dequantizeQ6K(
  quantized: GPUBuffer,
  numBlocks: number,
  options?: DequantOptions
): Promise<Tensor>;

/**
 * Run Q8_0 dequantization
 */
export declare function dequantizeQ8_0(
  quantized: GPUBuffer,
  numBlocks: number,
  options?: DequantOptions
): Promise<Tensor>;

/**
 * Record Q4_K_M dequantization (batched, no submit)
 */
export declare function recordDequantize(
  recorder: CommandRecorder,
  quantized: GPUBuffer,
  numBlocks: number,
  options?: DequantOptions
): Promise<Tensor>;
