import { getDevice } from '../../device.js';
import { acquireBuffer } from '../../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../../tensor.js';
import { WORKGROUP_SIZES } from '../constants.js';
import { dispatch, recordDispatch } from '../dispatch.js';
import { createPipeline, createUniformBufferWithView } from '../utils.js';
import { releaseUniformBuffer } from '../../uniform-cache.js';

export async function runBackwardKernel(
  opName,
  input,
  gradOutput,
  uniformSize,
  writeUniforms,
  options = {}
) {
  const device = getDevice();
  const { count, outputBuffer = null } = options;

  const bytesPerElement = dtypeBytes(gradOutput.dtype);
  const inferredCount = count ?? Math.floor(gradOutput.buffer.size / bytesPerElement);
  const pipeline = await createPipeline(opName, 'default');

  const outputSize = inferredCount * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, `${opName}_backward_output`);

  const uniformBuffer = createUniformBufferWithView(
    `${opName}_uniforms`,
    uniformSize,
    (view) => {
      writeUniforms(view, inferredCount);
    }
  );

  const bindGroup = device.createBindGroup({
    label: `${opName}_bind_group`,
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: gradOutput.buffer } },
      { binding: 3, resource: { buffer: outputBuf } },
    ],
  });

  const workgroups = Math.ceil(inferredCount / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, opName);

  releaseUniformBuffer(uniformBuffer);

  return createTensor(outputBuf, gradOutput.dtype, [...gradOutput.shape], `${opName}_output`);
}

export async function recordBackwardKernel(
  recorder,
  opName,
  input,
  gradOutput,
  uniformSize,
  writeUniforms,
  options = {}
) {
  const device = recorder.device;
  const { count, outputBuffer = null } = options;

  const bytesPerElement = dtypeBytes(gradOutput.dtype);
  const inferredCount = count ?? Math.floor(gradOutput.buffer.size / bytesPerElement);
  const pipeline = await createPipeline(opName, 'default');

  const outputSize = inferredCount * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, `${opName}_backward_output`);

  const uniformBuffer = createUniformBufferWithView(
    `${opName}_uniforms`,
    uniformSize,
    (view) => {
      writeUniforms(view, inferredCount);
    },
    recorder
  );

  const bindGroup = device.createBindGroup({
    label: `${opName}_bind_group`,
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: gradOutput.buffer } },
      { binding: 3, resource: { buffer: outputBuf } },
    ],
  });

  const workgroups = Math.ceil(inferredCount / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, opName);

  return createTensor(outputBuf, gradOutput.dtype, [...gradOutput.shape], `${opName}_output`);
}

export async function runMatmulTransposeA(A, B, M, N, K, options = {}) {
  const { alpha = 1.0, outputBuffer = null } = options;
  const device = getDevice();
  const outputSize = M * N * 4;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'matmul_transpose_a_output');

  const pipeline = await createPipeline('matmul_transpose_a', 'default');

  const uniformBuffer = createUniformBufferWithView(
    'matmul_transpose_a_uniforms',
    32,
    (view) => {
      view.setUint32(0, M, true);
      view.setUint32(4, N, true);
      view.setUint32(8, K, true);
      view.setFloat32(12, alpha, true);
    },
    null,
    device
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

  const TILE_SIZE = 16;
  const workgroups = [
    Math.ceil(M / TILE_SIZE),
    Math.ceil(N / TILE_SIZE),
    1,
  ];

  dispatch(device, pipeline, bindGroup, workgroups, 'matmul_transpose_a');
  releaseUniformBuffer(uniformBuffer);

  return createTensor(outputBuf, 'f32', [M, N], 'matmul_transpose_a_output');
}

export async function recordMatmulTransposeA(recorder, A, B, M, N, K, options = {}) {
  const { alpha = 1.0, outputBuffer = null } = options;
  const device = recorder.device;
  const outputSize = M * N * 4;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'matmul_transpose_a_output');

  const pipeline = await createPipeline('matmul_transpose_a', 'default');

  const uniformBuffer = createUniformBufferWithView(
    'matmul_transpose_a_uniforms',
    32,
    (view) => {
      view.setUint32(0, M, true);
      view.setUint32(4, N, true);
      view.setUint32(8, K, true);
      view.setFloat32(12, alpha, true);
    },
    recorder
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

  const TILE_SIZE = 16;
  const workgroups = [
    Math.ceil(M / TILE_SIZE),
    Math.ceil(N / TILE_SIZE),
    1,
  ];

  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'matmul_transpose_a');

  return createTensor(outputBuf, 'f32', [M, N], 'matmul_transpose_a_output');
}
