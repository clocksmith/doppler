/**
 * Fused GEMV + RMSNorm Kernel
 *
 * For decode (M=1), combines the down projection matmul with RMSNorm in a single kernel:
 * 1. Compute GEMV: C[1, N] = A[1, K] x B[K, N]  (down projection)
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
import { createTensor } from '../tensor.js';
import { getBuffer } from '../weight-buffer.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { getPipelineFast, createUniformBufferWithView } from './utils.js';
import { WORKGROUP_SIZES } from './constants.js';
import { getKernelThresholds } from '../../config/schema/kernel-thresholds.schema.js';
import { trace } from '../../debug/index.js';

/**
 * Select fused kernel variant based on output size
 *
 * - small: N <= WORKGROUP_SIZES.DEFAULT (one element per thread)
 * - medium: N > WORKGROUP_SIZES.DEFAULT (multiple elements per thread, single workgroup)
 * @param {number} N
 * @returns {string}
 */
export function selectMatmulRMSNormFusedVariant(N) {
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
 * @param {import('../tensor.js').Tensor} input - Input activation tensor [1, K]
 * @param {GPUBuffer | import('../weight-buffer.js').WeightBuffer} weight - Down projection weight buffer (GPUBuffer or WeightBuffer)
 * @param {GPUBuffer} normWeight - RMSNorm weight buffer [N]
 * @param {import('./fused_matmul_rmsnorm.js').MatmulRMSNormFusedOptions} options - Kernel options including N, K dimensions
 * @returns {Promise<import('../tensor.js').Tensor>} Output tensor [1, N] with normalized result
 */
export async function runMatmulRMSNormFused(
  input,
  weight,
  normWeight,
  options
) {
  const device = getDevice();
  const {
    N,
    K,
    eps = 1e-5,
    residual = null,
    outputBuffer = null,
    transposeB = true,  // Default: GGUF row-major weights
  } = options;

  const { colsPerWg } = getKernelThresholds().fusedMatmul;
  if (N > colsPerWg) {
    throw new Error(`[MatmulRMSNormFused] N=${N} exceeds colsPerWg=${colsPerWg}; kernel only supports single-workgroup RMSNorm.`);
  }

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
  /** @type {number} */
  let workgroups;
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
 * @param {import('../command-recorder.js').CommandRecorder} recorder
 * @param {import('../tensor.js').Tensor} input
 * @param {GPUBuffer | import('../weight-buffer.js').WeightBuffer} weight
 * @param {GPUBuffer} normWeight
 * @param {import('./fused_matmul_rmsnorm.js').MatmulRMSNormFusedOptions} options
 * @returns {Promise<import('../tensor.js').Tensor>}
 */
export async function recordMatmulRMSNormFused(
  recorder,
  input,
  weight,
  normWeight,
  options
) {
  const device = recorder.device;
  const {
    N,
    K,
    eps = 1e-5,
    residual = null,
    outputBuffer = null,
    transposeB = true,  // Default: GGUF row-major weights
  } = options;

  const { colsPerWg } = getKernelThresholds().fusedMatmul;
  if (N > colsPerWg) {
    throw new Error(`[MatmulRMSNormFused] N=${N} exceeds colsPerWg=${colsPerWg}; kernel only supports single-workgroup RMSNorm.`);
  }

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
  /** @type {number} */
  let workgroups;
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
 * - N <= colsPerWg (current WGSL RMSNorm reduction only valid for single workgroup)
 *
 * For N > 256, the parallelism loss from single-workgroup execution
 * outweighs the dispatch reduction benefit of fusion.
 * @param {number} M
 * @param {number} N
 * @returns {boolean}
 */
export function shouldUseFusedMatmulRMSNorm(M, N) {
  // Only beneficial for decode (M=1)
  if (M !== 1) {
    return false;
  }

  const { colsPerWg } = getKernelThresholds().fusedMatmul;
  if (N > colsPerWg) {
    return false;
  }

  return true;
}
