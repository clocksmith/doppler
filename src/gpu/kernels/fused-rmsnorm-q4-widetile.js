// JS dispatcher for fused_rmsnorm_q4_widetile.wgsl.
// Computes output[M, N] = rmsnorm(input[M, K]) * norm_weight @ dequant(B_q4k).
//
// Replaces a standalone rmsnorm dispatch + subsequent Q4K WideTile matmul
// with a single fused pass. Each caller (q_proj, k_proj, v_proj, gate_proj,
// up_proj) runs the rmsnorm independently inside the kernel — redundant
// across 3 q/k/v calls but negligible vs the matmul work, and the saved
// dispatch bubble on Dawn/Vulkan is the real prize.

import { getDevice } from '../device.js';
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { KernelBase } from './kernel-base.js';
import { createUniformBufferWithView } from './utils.js';
import { releaseUniformBuffer } from '../uniform-cache.js';
import { getBuffer } from '../weight-buffer.js';

const TILE_M = 4;
const TILE_N = 256;

class FusedRmsnormQ4WideTileKernel extends KernelBase {
  async getPipeline(variant, constants = null) {
    return this.getPipelineFor('matmul', variant, null, constants);
  }
  dispatch(pipeline, bindGroup, workgroups) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'fused_rmsnorm_q4_widetile');
  }
  record(recorder, pipeline, bindGroup, workgroups) {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, 'fused_rmsnorm_q4_widetile');
  }
}

let cachedKernel = null;
function getKernel(device) {
  if (!cachedKernel) cachedKernel = new FusedRmsnormQ4WideTileKernel(device);
  return cachedKernel;
}

function createUniform(device, recorder, params) {
  return createUniformBufferWithView(
    'fused_rmsnorm_q4_widetile_uniforms',
    32,
    (view) => {
      view.setUint32(0, params.M, true);
      view.setUint32(4, params.N, true);
      view.setUint32(8, params.K, true);
      view.setFloat32(12, params.alpha ?? 1.0, true);
      view.setUint32(16, Math.ceil(params.K / 256), true);
      view.setFloat32(20, params.eps ?? 1e-6, true);
    },
    recorder,
    device
  );
}

async function execute(recorder, input, weight, normWeight, options) {
  const { M, N, K, alpha = 1.0, eps = 1e-6, rmsNormOffset = false, outputBuffer = null } = options;

  if (!Number.isFinite(M) || !Number.isFinite(N) || !Number.isFinite(K)) {
    throw new Error('[FusedRmsnormQ4WideTile] M, N, K must be finite.');
  }
  if (input.dtype !== 'f32') {
    throw new Error('[FusedRmsnormQ4WideTile] input must be f32.');
  }

  const device = getDevice();
  if (!device) throw new Error('[FusedRmsnormQ4WideTile] No GPU device.');

  const kernel = getKernel(device);
  const pipeline = await kernel.getPipeline('q4_fused_rmsnorm_widetile', {
    RMS_NORM_OFFSET: rmsNormOffset,
  });

  const outputSize = M * N * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'fused_rmsnorm_q4_widetile_output');

  const uniform = createUniform(device, recorder, { M, N, K, alpha, eps });
  const weightBuf = getBuffer(weight);
  const normWeightBuf = getBuffer(normWeight);

  const bindGroup = device.createBindGroup({
    label: 'fused_rmsnorm_q4_widetile_bg',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: weightBuf } },
      { binding: 3, resource: { buffer: output } },
      { binding: 4, resource: { buffer: normWeightBuf } },
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
  return createTensor(output, 'f32', [M, N], 'fused_rmsnorm_q4_widetile_output');
}

export async function runFusedRmsnormQ4WideTile(input, weight, normWeight, options = {}) {
  return execute(null, input, weight, normWeight, options);
}

export async function recordFusedRmsnormQ4WideTile(recorder, input, weight, normWeight, options = {}) {
  return execute(recorder, input, weight, normWeight, options);
}
