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

export function createBackwardKernel(opName, spec) {
  const {
    uniformSize,
    writeUniforms,
    calcWorkgroups,
    outputBytes,
    outputShape,
    validate,
    dtype: specDtype,
    getDevice: useGetDevice,
  } = spec;

  async function run(...args) {
    const opts = args[args.length - 1];
    const inputs = args.slice(0, -1);

    if (validate) validate(opts);

    const { outputBuffer = null } = opts;
    const device = useGetDevice ? getDevice() : inputs[0].buffer.device;
    const outSize = outputBytes(opts);
    const outputBuf = outputBuffer || acquireBuffer(outSize, undefined, `${opName}_output`);

    const pipeline = await createPipeline(opName, 'default');

    const uniformBuffer = createUniformBufferWithView(
      `${opName}_uniforms`,
      uniformSize,
      (view) => writeUniforms(view, opts),
      null,
      device
    );

    const entries = [{ binding: 0, resource: { buffer: uniformBuffer } }];
    for (let i = 0; i < inputs.length; i++) {
      entries.push({ binding: i + 1, resource: { buffer: inputs[i].buffer } });
    }
    entries.push({ binding: inputs.length + 1, resource: { buffer: outputBuf } });

    const bindGroup = device.createBindGroup({
      label: `${opName}_bind_group`,
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });

    const workgroups = calcWorkgroups(opts);
    dispatch(device, pipeline, bindGroup, workgroups, opName);

    uniformBuffer.destroy();

    const dt = specDtype ? specDtype(opts, inputs) : (inputs[0] ? inputs[0].dtype : 'f32');
    return createTensor(outputBuf, dt, outputShape(opts), `${opName}_output`);
  }

  async function record(recorder, ...args) {
    const opts = args[args.length - 1];
    const inputs = args.slice(0, -1);

    if (validate) validate(opts);

    const { outputBuffer = null } = opts;
    const device = recorder.device;
    const outSize = outputBytes(opts);
    const outputBuf = outputBuffer || acquireBuffer(outSize, undefined, `${opName}_output`);

    const pipeline = await createPipeline(opName, 'default');

    const uniformBuffer = createUniformBufferWithView(
      `${opName}_uniforms`,
      uniformSize,
      (view) => writeUniforms(view, opts),
      recorder
    );

    const entries = [{ binding: 0, resource: { buffer: uniformBuffer } }];
    for (let i = 0; i < inputs.length; i++) {
      entries.push({ binding: i + 1, resource: { buffer: inputs[i].buffer } });
    }
    entries.push({ binding: inputs.length + 1, resource: { buffer: outputBuf } });

    const bindGroup = device.createBindGroup({
      label: `${opName}_bind_group`,
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });

    const workgroups = calcWorkgroups(opts);
    recordDispatch(recorder, pipeline, bindGroup, workgroups, opName);

    const dt = specDtype ? specDtype(opts, inputs) : (inputs[0] ? inputs[0].dtype : 'f32');
    return createTensor(outputBuf, dt, outputShape(opts), `${opName}_output`);
  }

  return { run, record };
}

export async function runMatmulBackwardDx(dY, W, M, K, N, options = {}) {
  const { alpha = 1.0, transposeB = false, outputBuffer = null } = options;
  const device = getDevice();
  const outputSize = M * K * 4;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'matmul_backward_dx_output');

  const pipeline = await createPipeline('matmul_backward', 'default');

  const uniformBuffer = createUniformBufferWithView(
    'matmul_backward_uniforms',
    32,
    (view) => {
      view.setUint32(0, M, true);
      view.setUint32(4, N, true);
      view.setUint32(8, K, true);
      view.setFloat32(12, alpha, true);
      view.setUint32(16, transposeB ? 1 : 0, true);
    },
    null,
    device
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

  const TILE_SIZE = 16;
  const workgroups = [
    Math.ceil(M / TILE_SIZE),
    Math.ceil(K / TILE_SIZE),
    1,
  ];

  dispatch(device, pipeline, bindGroup, workgroups, 'matmul_backward');
  releaseUniformBuffer(uniformBuffer);

  return createTensor(outputBuf, 'f32', [M, K], 'matmul_backward_dx_output');
}

export async function recordMatmulBackwardDx(recorder, dY, W, M, K, N, options = {}) {
  const { alpha = 1.0, transposeB = false, outputBuffer = null } = options;
  const device = recorder.device;
  const outputSize = M * K * 4;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'matmul_backward_dx_output');

  const pipeline = await createPipeline('matmul_backward', 'default');

  const uniformBuffer = createUniformBufferWithView(
    'matmul_backward_uniforms',
    32,
    (view) => {
      view.setUint32(0, M, true);
      view.setUint32(4, N, true);
      view.setUint32(8, K, true);
      view.setFloat32(12, alpha, true);
      view.setUint32(16, transposeB ? 1 : 0, true);
    },
    recorder
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

  const TILE_SIZE = 16;
  const workgroups = [
    Math.ceil(M / TILE_SIZE),
    Math.ceil(K / TILE_SIZE),
    1,
  ];

  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'matmul_backward');

  return createTensor(outputBuf, 'f32', [M, K], 'matmul_backward_dx_output');
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
