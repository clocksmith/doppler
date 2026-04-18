// JS dispatcher for fused_matmul_residual_f16.wgsl.
// Computes output[M, N] = (input @ weight) + residual in one compute pass.

import { getDevice } from '../device.js';
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { KernelBase } from './kernel-base.js';
import { createUniformBufferWithView } from './utils.js';
import { releaseUniformBuffer } from '../uniform-cache.js';
import { getBuffer } from '../weight-buffer.js';

const WORKGROUP_SIZE = 256;
const COLS_PER_WG = 64;

class FusedMatmulResidualKernel extends KernelBase {
  async getPipeline(variant, constants = null) {
    return this.getPipelineFor('matmul', variant, null, constants);
  }
  dispatch(pipeline, bindGroup, workgroups) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'fused_matmul_residual');
  }
  record(recorder, pipeline, bindGroup, workgroups) {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, 'fused_matmul_residual');
  }
}

let cachedKernel = null;
function getKernel(device) {
  if (!cachedKernel) cachedKernel = new FusedMatmulResidualKernel(device);
  return cachedKernel;
}

function createFusedMatmulResidualUniform(device, recorder, params) {
  return createUniformBufferWithView(
    'fused_matmul_residual_uniforms',
    32,
    (view) => {
      view.setUint32(0, params.M, true);
      view.setUint32(4, params.N, true);
      view.setUint32(8, params.K, true);
      view.setUint32(12, params.transposeB ? 1 : 0, true);
      view.setUint32(16, 0, true);
      view.setUint32(20, 0, true);
      view.setUint32(24, 0, true);
      view.setUint32(28, 0, true);
    },
    recorder,
    device
  );
}

async function executeFusedMatmulResidual(recorder, input, weight, residual, options) {
  const {
    M, N, K,
    transposeB = true,
    outputBuffer = null,
  } = options;

  if (!Number.isFinite(M) || !Number.isFinite(N) || !Number.isFinite(K)) {
    throw new Error('[FusedMatmulResidual] M, N, K must be finite numbers.');
  }
  if (input.dtype !== 'f16' || residual.dtype !== 'f16') {
    throw new Error('[FusedMatmulResidual] input and residual must be f16.');
  }

  const device = getDevice();
  if (!device) throw new Error('[FusedMatmulResidual] No GPU device.');

  const kernel = getKernel(device);
  const constants = { WORKGROUP_SIZE, COLS_PER_WG };
  const pipeline = await kernel.getPipeline('matmul_residual_tiled_f16', constants);

  const outputSize = M * N * 2;  // f16
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'fused_matmul_residual_output');

  const uniform = createFusedMatmulResidualUniform(device, recorder, { M, N, K, transposeB });
  const weightBuf = getBuffer(weight);

  const bindGroup = device.createBindGroup({
    label: 'fused_matmul_residual_bg',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: weightBuf } },
      { binding: 3, resource: { buffer: residual.buffer } },
      { binding: 4, resource: { buffer: output } },
    ],
  });

  const colTiles = Math.ceil(N / COLS_PER_WG);
  const workgroups = [colTiles, M, 1];

  if (recorder) {
    kernel.record(recorder, pipeline, bindGroup, workgroups);
  } else {
    kernel.dispatch(pipeline, bindGroup, workgroups);
  }

  if (uniform) releaseUniformBuffer(uniform);
  return createTensor(output, 'f16', [M, N], 'fused_matmul_residual_output');
}

export async function runFusedMatmulResidual(input, weight, residual, options = {}) {
  return executeFusedMatmulResidual(null, input, weight, residual, options);
}

export async function recordFusedMatmulResidual(recorder, input, weight, residual, options = {}) {
  return executeFusedMatmulResidual(recorder, input, weight, residual, options);
}
