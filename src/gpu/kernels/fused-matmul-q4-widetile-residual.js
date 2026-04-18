// JS dispatcher for fused_matmul_q4_widetile_residual.wgsl.
// Computes output[M, N] = (input @ dequant(B_q4k)) + residual in one pass.
//
// Same bind layout as q4_fused_widetile with an extra binding (5) for the
// residual tensor. Expected callers: attention/run.js o_proj site (with
// attn_residual as the residual tensor) and ffn/dense.js ffn_down site
// (with ffn_residual as the residual tensor).

import { getDevice } from '../device.js';
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { KernelBase } from './kernel-base.js';
import { createUniformBufferWithView } from './utils.js';
import { releaseUniformBuffer } from '../uniform-cache.js';
import { getBuffer } from '../weight-buffer.js';

const TILE_M = 4;
const TILE_N = 256;

class FusedQ4WideTileResidualKernel extends KernelBase {
  async getPipeline(variant, constants = null) {
    return this.getPipelineFor('matmul', variant, null, constants);
  }
  dispatch(pipeline, bindGroup, workgroups) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'fused_matmul_q4_widetile_residual');
  }
  record(recorder, pipeline, bindGroup, workgroups) {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, 'fused_matmul_q4_widetile_residual');
  }
}

let cachedKernel = null;
function getKernel(device) {
  if (!cachedKernel) cachedKernel = new FusedQ4WideTileResidualKernel(device);
  return cachedKernel;
}

function createUniform(device, recorder, params) {
  return createUniformBufferWithView(
    'fused_matmul_q4_widetile_residual_uniforms',
    32,
    (view) => {
      view.setUint32(0, params.M, true);
      view.setUint32(4, params.N, true);
      view.setUint32(8, params.K, true);
      view.setFloat32(12, params.alpha ?? 1.0, true);
      view.setUint32(16, Math.ceil(params.K / 256), true);
    },
    recorder,
    device
  );
}

async function execute(recorder, input, weight, residual, options) {
  const { M, N, K, alpha = 1.0, outputBuffer = null } = options;

  if (!Number.isFinite(M) || !Number.isFinite(N) || !Number.isFinite(K)) {
    throw new Error('[FusedMatmulQ4WideTileResidual] M, N, K must be finite.');
  }
  if (input.dtype !== 'f32') {
    throw new Error('[FusedMatmulQ4WideTileResidual] input must be f32.');
  }
  if (residual.dtype !== 'f32') {
    throw new Error('[FusedMatmulQ4WideTileResidual] residual must be f32.');
  }

  const device = getDevice();
  if (!device) throw new Error('[FusedMatmulQ4WideTileResidual] No GPU device.');

  const kernel = getKernel(device);
  const pipeline = await kernel.getPipeline('q4_fused_widetile_residual');

  const outputSize = M * N * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'fused_matmul_q4_widetile_residual_output');

  const uniform = createUniform(device, recorder, { M, N, K, alpha });
  const weightBuf = getBuffer(weight);

  const bindGroup = device.createBindGroup({
    label: 'fused_matmul_q4_widetile_residual_bg',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: weightBuf } },
      { binding: 3, resource: { buffer: output } },
      { binding: 4, resource: { buffer: residual.buffer } },
    ],
  });

  const workgroupsX = Math.ceil(N / TILE_N);
  const workgroupsY = Math.ceil(M / TILE_M);
  const workgroups = [workgroupsX, workgroupsY, 1];

  if (recorder) {
    kernel.record(recorder, pipeline, bindGroup, workgroups);
  } else {
    kernel.dispatch(pipeline, bindGroup, workgroups);
  }

  if (uniform) releaseUniformBuffer(uniform);
  return createTensor(output, 'f32', [M, N], 'fused_matmul_q4_widetile_residual_output');
}

export async function runFusedMatmulQ4WideTileResidual(input, weight, residual, options = {}) {
  return execute(null, input, weight, residual, options);
}

export async function recordFusedMatmulQ4WideTileResidual(recorder, input, weight, residual, options = {}) {
  return execute(recorder, input, weight, residual, options);
}
