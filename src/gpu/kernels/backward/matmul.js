import { getDevice } from '../../device.js';
import { createTensor } from '../../tensor.js';
import { releaseUniformBuffer } from '../../uniform-cache.js';
import { acquireBuffer } from '../../../memory/buffer-pool.js';
import { dispatch, recordDispatch } from '../dispatch.js';
import { createPipeline } from '../pipeline-cache.js';
import { createUniformBufferWithView } from '../uniform-utils.js';

const TILE_SIZE = 16;

export function resolveMatmulBackwardDxVariant(weight) {
  const dtype = String(weight?.dtype || 'f32').toLowerCase();
  if (dtype === 'f16') return 'f16_weight';
  if (dtype === 'f32') return 'default';
  if (dtype === 'q4k') return 'q4k_weight';
  throw new Error(`matmul backward dX does not support weight dtype "${dtype}".`);
}

function resolveMatmulBackwardDxDispatch(weight, K, transposeB) {
  const variant = resolveMatmulBackwardDxVariant(weight);
  if (variant === 'q4k_weight' && !transposeB) {
    throw new Error('matmul backward dX requires transposeB=true for row-wise Q4K weights [N,K].');
  }
  return {
    variant,
    numBlocksPerRow: variant === 'q4k_weight' ? Math.ceil(K / 256) : 0,
  };
}

function writeMatmulBackwardUniforms(view, M, N, K, alpha, transposeB, numBlocksPerRow) {
  view.setUint32(0, M, true);
  view.setUint32(4, N, true);
  view.setUint32(8, K, true);
  view.setFloat32(12, alpha, true);
  view.setUint32(16, transposeB ? 1 : 0, true);
  view.setUint32(20, numBlocksPerRow, true);
}

async function executeMatmulBackwardDx(recorder, dY, W, M, K, N, options) {
  const { alpha = 1.0, transposeB = false, outputBuffer = null } = options;
  const device = recorder?.device ?? getDevice();
  const outputBuf = outputBuffer || acquireBuffer(M * K * 4, undefined, 'matmul_backward_dx_output');
  const dispatchConfig = resolveMatmulBackwardDxDispatch(W, K, transposeB);
  const pipeline = await createPipeline('matmul_backward', dispatchConfig.variant);
  const uniformBuffer = createUniformBufferWithView(
    'matmul_backward_uniforms',
    32,
    (view) => writeMatmulBackwardUniforms(
      view,
      M,
      N,
      K,
      alpha,
      transposeB,
      dispatchConfig.numBlocksPerRow
    ),
    recorder,
    recorder ? undefined : device
  );
  const bindGroup = device.createBindGroup({
    label: 'matmul_backward_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: dY.buffer } },
      { binding: 2, resource: { buffer: W.buffer } },
      { binding: 3, resource: { buffer: outputBuf } },
    ],
  });
  const workgroups = [Math.ceil(M / TILE_SIZE), Math.ceil(K / TILE_SIZE), 1];
  if (recorder) {
    recordDispatch(recorder, pipeline, bindGroup, workgroups, 'matmul_backward');
  } else {
    dispatch(device, pipeline, bindGroup, workgroups, 'matmul_backward');
    releaseUniformBuffer(uniformBuffer);
  }
  return createTensor(outputBuf, 'f32', [M, K], 'matmul_backward_dx_output');
}

export function runMatmulBackwardDx(dY, W, M, K, N, options = {}) {
  return executeMatmulBackwardDx(null, dY, W, M, K, N, options);
}

export function recordMatmulBackwardDx(recorder, dY, W, M, K, N, options = {}) {
  return executeMatmulBackwardDx(recorder, dY, W, M, K, N, options);
}

function writeMatmulTransposeAUniforms(view, M, N, K, alpha) {
  view.setUint32(0, M, true);
  view.setUint32(4, N, true);
  view.setUint32(8, K, true);
  view.setFloat32(12, alpha, true);
}

async function executeMatmulTransposeA(recorder, A, B, M, N, K, options) {
  const { alpha = 1.0, outputBuffer = null } = options;
  const device = recorder?.device ?? getDevice();
  const outputBuf = outputBuffer || acquireBuffer(M * N * 4, undefined, 'matmul_transpose_a_output');
  const pipeline = await createPipeline('matmul_transpose_a', 'default');
  const uniformBuffer = createUniformBufferWithView(
    'matmul_transpose_a_uniforms',
    32,
    (view) => writeMatmulTransposeAUniforms(view, M, N, K, alpha),
    recorder,
    recorder ? undefined : device
  );
  const bindGroup = device.createBindGroup({
    label: 'matmul_transpose_a_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: A.buffer } },
      { binding: 2, resource: { buffer: B.buffer } },
      { binding: 3, resource: { buffer: outputBuf } },
    ],
  });
  const workgroups = [Math.ceil(M / TILE_SIZE), Math.ceil(N / TILE_SIZE), 1];
  if (recorder) {
    recordDispatch(recorder, pipeline, bindGroup, workgroups, 'matmul_transpose_a');
  } else {
    dispatch(device, pipeline, bindGroup, workgroups, 'matmul_transpose_a');
    releaseUniformBuffer(uniformBuffer);
  }
  return createTensor(outputBuf, 'f32', [M, N], 'matmul_transpose_a_output');
}

export function runMatmulTransposeA(A, B, M, N, K, options = {}) {
  return executeMatmulTransposeA(null, A, B, M, N, K, options);
}

export function recordMatmulTransposeA(recorder, A, B, M, N, K, options = {}) {
  return executeMatmulTransposeA(recorder, A, B, M, N, K, options);
}
