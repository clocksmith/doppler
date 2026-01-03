/**
 * Kernel Benchmark Functions
 *
 * Individual benchmark implementations for different kernel types.
 * Each benchmark creates test buffers, runs warmup iterations, and measures performance.
 */

import type {
  InputSizes,
  WorkgroupSize,
  TuneResult,
  KernelCapabilities,
} from './types.js';

/**
 * Benchmark a compute pipeline with given workgroups
 * @param device - GPU device
 * @param pipeline - Compute pipeline to benchmark
 * @param bindGroup - Bind group for the pipeline
 * @param workgroups - Number of workgroups [x, y, z]
 * @param warmup - Number of warmup iterations
 * @param iterations - Number of timed iterations
 * @returns Average time in milliseconds
 */
export function benchmarkPipeline(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  workgroups: [number, number, number],
  warmup: number,
  iterations: number
): Promise<number>;

/**
 * Create compute pipeline from shader source
 * @param device - GPU device
 * @param shaderSource - WGSL shader source code
 * @param entryPoint - Entry point function name
 * @returns Compute pipeline
 */
export function createComputePipeline(
  device: GPUDevice,
  shaderSource: string,
  entryPoint: string
): Promise<GPUComputePipeline>;

/**
 * Tune matmul kernel
 * @param device - GPU device
 * @param inputSizes - Matrix dimensions
 * @param candidates - Workgroup size candidates
 * @param warmup - Warmup iterations
 * @param iterations - Timed iterations
 * @param capabilities - Kernel capabilities
 * @returns Best tuning result
 */
export function tuneMatmul(
  device: GPUDevice,
  inputSizes: InputSizes,
  candidates: WorkgroupSize[],
  warmup: number,
  iterations: number,
  capabilities: KernelCapabilities | null
): Promise<TuneResult>;

/**
 * Create matmul shader with specified workgroup size
 */
export function createMatmulShader(wgX: number, wgY: number): string;

/**
 * Tune attention kernel
 */
export function tuneAttention(
  device: GPUDevice,
  inputSizes: InputSizes,
  candidates: WorkgroupSize[],
  warmup: number,
  iterations: number,
  capabilities: KernelCapabilities | null
): Promise<TuneResult>;

/**
 * Create attention shader with specified workgroup size
 */
export function createAttentionShader(wgSize: number): string;

/**
 * Tune softmax kernel
 */
export function tuneSoftmax(
  device: GPUDevice,
  inputSizes: InputSizes,
  candidates: WorkgroupSize[],
  warmup: number,
  iterations: number,
  capabilities: KernelCapabilities | null
): Promise<TuneResult>;

/**
 * Create softmax shader with specified workgroup size
 */
export function createSoftmaxShader(wgSize: number): string;

/**
 * Tune RMSNorm kernel
 */
export function tuneRMSNorm(
  device: GPUDevice,
  inputSizes: InputSizes,
  candidates: WorkgroupSize[],
  warmup: number,
  iterations: number,
  capabilities: KernelCapabilities | null
): Promise<TuneResult>;

/**
 * Create RMSNorm shader with specified workgroup size
 */
export function createRMSNormShader(wgSize: number): string;

/**
 * Tune dequantization kernel
 */
export function tuneDequant(
  device: GPUDevice,
  inputSizes: InputSizes,
  candidates: WorkgroupSize[],
  warmup: number,
  iterations: number,
  capabilities: KernelCapabilities | null
): Promise<TuneResult>;

/**
 * Create dequant shader with specified workgroup size
 */
export function createDequantShader(wgSize: number): string;

/**
 * Generic tuning for unknown kernels - returns sensible defaults
 */
export function tuneGeneric(
  capabilities: KernelCapabilities | null
): TuneResult;
