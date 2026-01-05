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
import { acquireBuffer } from '../buffer-pool.js';
import type { CommandRecorder } from '../command-recorder.js';
import { createTensor, type Tensor } from '../tensor.js';
import { type WeightBuffer, getBuffer } from '../weight-buffer.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { getPipelineFast, createUniformBufferWithView } from './utils.js';
import { WORKGROUP_SIZES } from './constants.js';
import { getKernelThresholds } from '../../config/schema/kernel-thresholds.schema.js';
import type { OutputBufferOptions } from './types.js';
import { trace } from '../../debug/index.js';

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
  /**
   * Whether weight matrix is stored transposed.
   * - true: weight is [N,K] (row-major/SafeTensors), needs transpose access
   * - false: weight is [K,N] (column-major/pre-transposed), direct access
   * Default: true (matches GGUF convention)
   */
  transposeB?: boolean;
}

/**
 * Select fused kernel variant based on output size
 *
 * - small: N <= WORKGROUP_SIZES.DEFAULT (one element per thread)
 * - medium: N > WORKGROUP_SIZES.DEFAULT (multiple elements per thread, single workgroup)
 */
export function selectMatmulRMSNormFusedVariant(N: number): string {
  if (N <= WORKGROUP_SIZES.DEFAULT) {
    return 'small';
  }
  return 'medium';
}

/**
 * Run fused GEMV + RMSNorm
 *
 * Combines down projection matmul (M=1) with RMSNorm in a single kernel.
 * Use this for the post-FFN normalization path during decode.
 *
 * @param input - Input activation tensor [1, K]
 * @param weight - Down projection weight buffer (GPUBuffer or WeightBuffer)
 * @param normWeight - RMSNorm weight buffer [N]
 * @param options - Kernel options including N, K dimensions
 * @returns Output tensor [1, N] with normalized result
 */
export async function runMatmulRMSNormFused(
  input: Tensor,
  weight: GPUBuffer | WeightBuffer,
  normWeight: GPUBuffer,
  options: MatmulRMSNormFusedOptions
): Promise<Tensor> {
  const device = getDevice();
  const {
    N,
    K,
    eps = 1e-5,
    residual = null,
    outputBuffer = null,
    transposeB = true,  // Default: GGUF row-major weights
  } = options;

  const weightBuffer = getBuffer(weight);

  // Select variant based on output size
  const variant = selectMatmulRMSNormFusedVariant(N);

  trace.kernels(`MatmulRMSNormFused: N=${N}, K=${K}, variant=${variant}, hasResidual=${!!residual}, transposeB=${transposeB}`);

  const pipeline = await getPipelineFast('fused_matmul_rmsnorm', variant);

  // Output buffer: [1, N] floats
  const outputSize = N * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'matmul_rmsnorm_fused_output');

  // Create uniform buffer (8 u32/f32 = 32 bytes, padded for alignment)
  const uniformBuffer = createUniformBufferWithView(
    'matmul_rmsnorm_fused_uniforms',
    32,
    (view) => {
      view.setUint32(0, N, true);
      view.setUint32(4, K, true);
      view.setFloat32(8, eps, true);
      view.setUint32(12, residual ? 1 : 0, true);
      view.setUint32(16, transposeB ? 1 : 0, true);
      // Padding bytes 20-31 are zero-initialized
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
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: weightBuffer } },
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
    workgroups = Math.ceil(N / getKernelThresholds().fusedMatmul.colsPerWg);
  }

  dispatch(device, pipeline, bindGroup, workgroups, 'matmul_rmsnorm_fused');

  // Cleanup
  uniformBuffer.destroy();
  if (!residual) residualBuffer.destroy();

  // Output dtype matches input dtype
  return createTensor(output, input.dtype, [1, N], 'matmul_rmsnorm_fused_output');
}

/**
 * Record fused GEMV + RMSNorm (batched, no submit)
 *
 * Use this for the command recording path in the inference pipeline.
 */
export async function recordMatmulRMSNormFused(
  recorder: CommandRecorder,
  input: Tensor,
  weight: GPUBuffer | WeightBuffer,
  normWeight: GPUBuffer,
  options: MatmulRMSNormFusedOptions
): Promise<Tensor> {
  const device = recorder.device;
  const {
    N,
    K,
    eps = 1e-5,
    residual = null,
    outputBuffer = null,
    transposeB = true,  // Default: GGUF row-major weights
  } = options;

  const weightBuffer = getBuffer(weight);

  // Select variant
  const variant = selectMatmulRMSNormFusedVariant(N);

  trace.kernels(`recordMatmulRMSNormFused: N=${N}, K=${K}, variant=${variant}, hasResidual=${!!residual}, transposeB=${transposeB}`);

  const pipeline = await getPipelineFast('fused_matmul_rmsnorm', variant);

  // Output buffer
  const outputSize = N * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'matmul_rmsnorm_fused_output');

  // Uniform buffer via recorder (8 u32/f32 = 32 bytes, padded for alignment)
  const uniformBuffer = createUniformBufferWithView(
    'matmul_rmsnorm_fused_uniforms',
    32,
    (view) => {
      view.setUint32(0, N, true);
      view.setUint32(4, K, true);
      view.setFloat32(8, eps, true);
      view.setUint32(12, residual ? 1 : 0, true);
      view.setUint32(16, transposeB ? 1 : 0, true);
      // Padding bytes 20-31 are zero-initialized
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
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: weightBuffer } },
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
    workgroups = Math.ceil(N / getKernelThresholds().fusedMatmul.colsPerWg);
  }

  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'matmul_rmsnorm_fused');

  // Track placeholder for cleanup
  if (!residual) {
    recorder.trackTemporaryBuffer(residualBuffer);
  }

  // Output dtype matches input dtype
  return createTensor(output, input.dtype, [1, N], 'matmul_rmsnorm_fused_output');
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
  // Medium variant handles N up to maxMediumN (e.g., Gemma 3 hiddenSize=1152)
  if (N > getKernelThresholds().fusedMatmul.maxMediumN) {
    return false;
  }

  return true;
}
