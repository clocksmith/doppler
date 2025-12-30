/**
 * Fused GEMV + RMSNorm Kernel
 *
 * For decode (M=1), combines the down projection matmul with RMSNorm in a single kernel:
 * 1. Compute GEMV: C[1, N] = A[1, K] × B[K, N]  (down projection)
 * 2. Compute RMSNorm on C: output = C / sqrt(mean(C^2) + eps) * weight
 * 3. Optional residual: output = output + residual
 *
 * Benefits:
 * - Single GPU dispatch instead of 2
 * - No intermediate buffer for matmul output
 * - Better cache locality
 *
 * Expected speedup: 1.2-1.5x for post-FFN normalization path
 */

import { getDevice } from '../device.js';
import { setBufferDtype } from '../buffer-dtypes.js';
import { acquireBuffer } from '../buffer-pool.js';
import type { CommandRecorder } from '../command-recorder.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView } from './utils.js';
import type { OutputBufferOptions } from './types.js';

const DEBUG_KERNELS = typeof window !== 'undefined'
  ? Boolean((window as unknown as { DOPPLER_DEBUG_KERNELS?: boolean }).DOPPLER_DEBUG_KERNELS)
  : false;

/** Kernel constants matching WGSL */
const WG_SIZE = 256;
const COLS_PER_WG = 4;  // For multi-workgroup variant
const MAX_MEDIUM_N = 4096;

/** Fused MatmulRMSNorm kernel options */
export interface MatmulRMSNormFusedOptions extends OutputBufferOptions {
  /** Output dimension N (hiddenSize) */
  N: number;
  /** Input dimension K (intermediateSize) */
  K: number;
  /** RMSNorm epsilon (default: 1e-5) */
  eps?: number;
  /** Optional residual buffer to add to output */
  residual?: GPUBuffer | null;
}

/**
 * Select fused kernel variant based on output size
 *
 * - small: N <= 256 (one element per thread)
 * - medium: 256 < N <= 4096 (multiple elements per thread, single workgroup)
 * - default: N > 4096 (multi-workgroup, but RMSNorm is incomplete - avoid)
 */
export function selectMatmulRMSNormFusedVariant(N: number): string {
  if (N <= WG_SIZE) {
    return 'small';  // Single workgroup, one element per thread
  }
  if (N <= MAX_MEDIUM_N) {
    return 'medium';  // Single workgroup, multiple elements per thread
  }
  // For very large N, fall back to default (incomplete RMSNorm)
  // In practice, should not use fused kernel for N > 4096
  return 'default';
}

/**
 * Run fused GEMV + RMSNorm
 *
 * Combines down projection matmul (M=1) with RMSNorm in a single kernel.
 * Use this for the post-FFN normalization path during decode.
 *
 * @param input - Input activation buffer [1, K]
 * @param weight - Down projection weight buffer [K, N] (row-major)
 * @param normWeight - RMSNorm weight buffer [N]
 * @param options - Kernel options including N, K dimensions
 * @returns Output buffer [1, N] with normalized result
 */
export async function runMatmulRMSNormFused(
  input: GPUBuffer,
  weight: GPUBuffer,
  normWeight: GPUBuffer,
  options: MatmulRMSNormFusedOptions
): Promise<GPUBuffer> {
  const device = getDevice();
  const {
    N,
    K,
    eps = 1e-5,
    residual = null,
    outputBuffer = null,
  } = options;

  // Select variant based on output size
  const variant = selectMatmulRMSNormFusedVariant(N);

  if (DEBUG_KERNELS) {
    console.log(`[MatmulRMSNormFused] N=${N}, K=${K}, variant=${variant}, hasResidual=${!!residual}`);
  }

  const pipeline = await createPipeline('fused_matmul_rmsnorm', variant);

  // Output buffer: [1, N] floats
  const outputSize = N * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'matmul_rmsnorm_fused_output');

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'matmul_rmsnorm_fused_uniforms',
    16,
    (view) => {
      view.setUint32(0, N, true);
      view.setUint32(4, K, true);
      view.setFloat32(8, eps, true);
      view.setUint32(12, residual ? 1 : 0, true);
    },
    null,
    device
  );

  // Create placeholder for residual if not provided
  const residualBuffer = residual || device.createBuffer({
    label: 'matmul_rmsnorm_residual_placeholder',
    size: 4,
    usage: GPUBufferUsage.STORAGE,
  });

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'matmul_rmsnorm_fused_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: weight } },
      { binding: 3, resource: { buffer: normWeight } },
      { binding: 4, resource: { buffer: output } },
      { binding: 5, resource: { buffer: residualBuffer } },
    ],
  });

  // Calculate workgroups
  let workgroups: number;
  if (variant === 'small' || variant === 'medium') {
    workgroups = 1;  // Single workgroup for small/medium N
  } else {
    workgroups = Math.ceil(N / COLS_PER_WG);
  }

  dispatch(device, pipeline, bindGroup, workgroups, 'matmul_rmsnorm_fused');

  // Cleanup
  uniformBuffer.destroy();
  if (!residual) residualBuffer.destroy();

  setBufferDtype(output, 'f32');
  return output;
}

/**
 * Record fused GEMV + RMSNorm (batched, no submit)
 *
 * Use this for the command recording path in the inference pipeline.
 */
export async function recordMatmulRMSNormFused(
  recorder: CommandRecorder,
  input: GPUBuffer,
  weight: GPUBuffer,
  normWeight: GPUBuffer,
  options: MatmulRMSNormFusedOptions
): Promise<GPUBuffer> {
  const device = recorder.device;
  const {
    N,
    K,
    eps = 1e-5,
    residual = null,
    outputBuffer = null,
  } = options;

  // Select variant
  const variant = selectMatmulRMSNormFusedVariant(N);

  if (DEBUG_KERNELS) {
    console.log(`[recordMatmulRMSNormFused] N=${N}, K=${K}, variant=${variant}, hasResidual=${!!residual}`);
  }

  const pipeline = await createPipeline('fused_matmul_rmsnorm', variant);

  // Output buffer
  const outputSize = N * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'matmul_rmsnorm_fused_output');

  // Uniform buffer via recorder
  const uniformBuffer = createUniformBufferWithView(
    'matmul_rmsnorm_fused_uniforms',
    16,
    (view) => {
      view.setUint32(0, N, true);
      view.setUint32(4, K, true);
      view.setFloat32(8, eps, true);
      view.setUint32(12, residual ? 1 : 0, true);
    },
    recorder
  );

  // Placeholder for residual
  const residualBuffer = residual || device.createBuffer({
    label: 'matmul_rmsnorm_residual_placeholder',
    size: 4,
    usage: GPUBufferUsage.STORAGE,
  });

  // Bind group
  const bindGroup = device.createBindGroup({
    label: 'matmul_rmsnorm_fused_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: weight } },
      { binding: 3, resource: { buffer: normWeight } },
      { binding: 4, resource: { buffer: output } },
      { binding: 5, resource: { buffer: residualBuffer } },
    ],
  });

  // Calculate workgroups
  let workgroups: number;
  if (variant === 'small' || variant === 'medium') {
    workgroups = 1;  // Single workgroup for small/medium N
  } else {
    workgroups = Math.ceil(N / COLS_PER_WG);
  }

  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'matmul_rmsnorm_fused');

  // Track placeholder for cleanup
  if (!residual) {
    recorder.trackTemporaryBuffer(residualBuffer);
  }

  setBufferDtype(output, 'f32');
  return output;
}

/**
 * Check if fused kernel should be used for given dimensions
 *
 * The fused kernel is beneficial when:
 * - M = 1 (decode, not prefill)
 * - N <= 256 (small variant works well, medium variant has parallelism issues)
 *
 * For N > 256, the parallelism loss from single-workgroup execution
 * outweighs the dispatch reduction benefit of fusion.
 */
export function shouldUseFusedMatmulRMSNorm(M: number, N: number): boolean {
  // Only beneficial for decode (M=1)
  if (M !== 1) {
    return false;
  }

  // Enable for small and medium N where single-workgroup is efficient
  // Medium variant handles N up to 4096 (e.g., Gemma 3 hiddenSize=1152)
  if (N > MAX_MEDIUM_N) {
    return false;
  }

  return true;
}
