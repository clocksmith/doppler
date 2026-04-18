// JS dispatcher for fused_rmsnorm_matmul_tiled_f16.wgsl.
// Computes output[M, N] = rmsnorm(input[M, K]) @ weight in one compute pass.

import { getDevice } from '../device.js';
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { KernelBase } from './kernel-base.js';
import { createUniformBufferWithView } from './utils.js';
import { releaseUniformBuffer } from '../uniform-cache.js';
import { dispatch as directDispatch } from './dispatch.js';
import { getBuffer } from '../weight-buffer.js';

const WORKGROUP_SIZE = 256;
const COLS_PER_WG = 64;

class FusedRmsnormMatmulKernel extends KernelBase {
  async getPipeline(variant, constants = null) {
    return this.getPipelineFor('matmul', variant, null, constants);
  }
  dispatch(pipeline, bindGroup, workgroups) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'fused_rmsnorm_matmul');
  }
  record(recorder, pipeline, bindGroup, workgroups) {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, 'fused_rmsnorm_matmul');
  }
}

let cachedKernel = null;
function getKernel(device) {
  if (!cachedKernel) cachedKernel = new FusedRmsnormMatmulKernel(device);
  return cachedKernel;
}

function createFusedRmsnormMatmulUniform(device, recorder, params) {
  return createUniformBufferWithView(
    'fused_rmsnorm_matmul_uniforms',
    32,
    (view) => {
      view.setUint32(0, params.M, true);
      view.setUint32(4, params.N, true);
      view.setUint32(8, params.K, true);
      view.setFloat32(12, params.eps, true);
      view.setUint32(16, params.transposeB ? 1 : 0, true);
      view.setUint32(20, 0, true);
      view.setUint32(24, 0, true);
      view.setUint32(28, 0, true);
    },
    recorder,
    device
  );
}

async function executeFusedRmsnormMatmul(recorder, input, matmulWeight, normWeight, options) {
  const {
    M,
    N,
    K,
    eps = 1e-6,
    transposeB = true,
    rmsNormOffset = false,
    weightIsF16 = false,
    outputBuffer = null,
  } = options;

  if (!Number.isFinite(M) || !Number.isFinite(N) || !Number.isFinite(K)) {
    throw new Error('[FusedRmsnormMatmul] M, N, K must be finite numbers.');
  }
  if (input.dtype !== 'f16') {
    throw new Error('[FusedRmsnormMatmul] input must be f16.');
  }

  const device = getDevice();
  if (!device) throw new Error('[FusedRmsnormMatmul] No GPU device.');

  const kernel = getKernel(device);
  const constants = {
    WORKGROUP_SIZE,
    COLS_PER_WG,
    RMS_NORM_OFFSET: rmsNormOffset,
    WEIGHT_IS_F16: weightIsF16,
  };
  const pipeline = await kernel.getPipeline('rmsnorm_matmul_tiled_f16', constants);

  const outputSize = M * N * 2;  // f16
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'fused_rmsnorm_matmul_output');

  const uniform = createFusedRmsnormMatmulUniform(device, recorder, {
    M, N, K, eps, transposeB,
  });

  const matmulWeightBuf = getBuffer(matmulWeight);
  const normWeightBuf = getBuffer(normWeight);

  const bindGroup = device.createBindGroup({
    label: 'fused_rmsnorm_matmul_bg',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: matmulWeightBuf } },
      { binding: 3, resource: { buffer: normWeightBuf } },
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

  if (uniform) {
    releaseUniformBuffer(uniform);
  }

  return createTensor(output, 'f16', [M, N], 'fused_rmsnorm_matmul_output');
}

export async function runFusedRmsnormMatmul(input, matmulWeight, normWeight, options = {}) {
  return executeFusedRmsnormMatmul(null, input, matmulWeight, normWeight, options);
}

export async function recordFusedRmsnormMatmul(recorder, input, matmulWeight, normWeight, options = {}) {
  return executeFusedRmsnormMatmul(recorder, input, matmulWeight, normWeight, options);
}
