/**
 * Fused GEMV + Residual Kernel
 *
 * For decode (M=1), combines output projection matmul with residual add in a single kernel:
 * C[N] = A[K] * B^T[K,N] + residual[N]
 *
 * Benefits:
 * - Single GPU dispatch instead of 2
 * - No intermediate buffer for matmul output
 * - Better cache locality
 *
 * Expected speedup: eliminates 1 dispatch barrier per layer (16 barriers for 16-layer model)
 */

import { getDevice } from '../device.js';
import { acquireBuffer } from '../buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { getBuffer } from '../weight-buffer.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { getPipelineFast, createUniformBufferWithView } from './utils.js';
import { trace } from '../../debug/index.js';

/**
 * Check if fused GEMV+residual should be used.
 *
 * Only use for decode (M=1) where GEMV kernel is applicable.
 * @param {number} M
 * @returns {boolean}
 */
export function shouldUseFusedMatmulResidual(M) {
  return M === 1;
}

/**
 * Run fused GEMV + Residual
 *
 * Combines output projection matmul (M=1) with residual add in a single kernel.
 * Use this for the attention output path during decode.
 *
 * @param {import('../tensor.js').Tensor} input - Input activation tensor [1, K] (attention output before o_proj)
 * @param {GPUBuffer | import('../weight-buffer.js').WeightBuffer} weight - Output projection weight buffer (GPUBuffer or WeightBuffer)
 * @param {import('../tensor.js').Tensor} residual - Residual tensor [1, N] (original input to add)
 * @param {import('./fused_matmul_residual.js').MatmulResidualFusedOptions} options - Kernel options including N, K dimensions
 * @returns {Promise<import('../tensor.js').Tensor>} Output tensor [1, N] with projected + residual result
 */
export async function runMatmulResidualFused(
  input,
  weight,
  residual,
  options
) {
  const device = getDevice();
  const {
    N,
    K,
    alpha = 1.0,
    outputBuffer = null,
  } = options;

  const weightBuffer = getBuffer(weight);
  /** @type {import('../tensor.js').TensorDtype} */
  const outputDtype = input.dtype;

  trace.kernels(`MatmulResidualFused: N=${N}, K=${K}, alpha=${alpha}, dtype=${outputDtype}`);

  const pipeline = await getPipelineFast('fused_matmul_residual', 'default');

  const output = outputBuffer || acquireBuffer(N * dtypeBytes(outputDtype), undefined, 'matmul_residual_output');

  // Create uniform buffer (same layout as matmul_gemv)
  const uniformBuffer = createUniformBufferWithView(
    'matmul_residual_uniforms',
    32,  // 8 u32s
    (view) => {
      view.setUint32(0, 1, true);         // M = 1 (decode)
      view.setUint32(4, N, true);         // N (output dimension)
      view.setUint32(8, K, true);         // K (input dimension)
      view.setFloat32(12, alpha, true);   // alpha
      view.setUint32(16, 1, true);        // transpose_b = 1
      view.setUint32(20, 0, true);        // _pad0
      view.setUint32(24, 0, true);        // _pad1
      view.setUint32(28, 0, true);        // _pad2
    },
    null,
    device
  );

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'matmul_residual_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: weightBuffer } },
      { binding: 3, resource: { buffer: output } },
      { binding: 4, resource: { buffer: residual.buffer } },
    ],
  });

  // One workgroup per output element
  const workgroups = N;
  dispatch(device, pipeline, bindGroup, workgroups, 'matmul_residual_fused');

  uniformBuffer.destroy();

  return createTensor(output, outputDtype, [1, N], 'matmul_residual_output');
}

/**
 * Record fused GEMV + Residual (batched, no submit)
 * @param {import('../command-recorder.js').CommandRecorder} recorder
 * @param {import('../tensor.js').Tensor} input
 * @param {GPUBuffer | import('../weight-buffer.js').WeightBuffer} weight
 * @param {import('../tensor.js').Tensor} residual
 * @param {import('./fused_matmul_residual.js').MatmulResidualFusedOptions} options
 * @returns {Promise<import('../tensor.js').Tensor>}
 */
export async function recordMatmulResidualFused(
  recorder,
  input,
  weight,
  residual,
  options
) {
  const device = recorder.device;
  const {
    N,
    K,
    alpha = 1.0,
    outputBuffer = null,
  } = options;

  const weightBuffer = getBuffer(weight);
  /** @type {import('../tensor.js').TensorDtype} */
  const outputDtype = input.dtype;

  const pipeline = await getPipelineFast('fused_matmul_residual', 'default');

  const output = outputBuffer || acquireBuffer(N * dtypeBytes(outputDtype), undefined, 'matmul_residual_output');

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'matmul_residual_uniforms',
    32,
    (view) => {
      view.setUint32(0, 1, true);         // M = 1
      view.setUint32(4, N, true);         // N
      view.setUint32(8, K, true);         // K
      view.setFloat32(12, alpha, true);   // alpha
      view.setUint32(16, 1, true);        // transpose_b = 1
      view.setUint32(20, 0, true);        // _pad0
      view.setUint32(24, 0, true);        // _pad1
      view.setUint32(28, 0, true);        // _pad2
    },
    recorder
  );

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'matmul_residual_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: weightBuffer } },
      { binding: 3, resource: { buffer: output } },
      { binding: 4, resource: { buffer: residual.buffer } },
    ],
  });

  // One workgroup per output element
  const workgroups = N;
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'matmul_residual_fused');

  return createTensor(output, outputDtype, [1, N], 'matmul_residual_output');
}
