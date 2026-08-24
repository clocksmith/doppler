import { getDevice } from '../../device.js';
import { createTensor, dtypeBytes } from '../../tensor.js';
import { releaseUniformBuffer } from '../../uniform-cache.js';
import { acquireBuffer } from '../../../memory/buffer-pool.js';
import { WORKGROUP_SIZES } from '../constants.js';
import { dispatch, recordDispatch } from '../dispatch.js';
import { createPipeline } from '../pipeline-cache.js';
import { createUniformBufferWithView } from '../uniform-utils.js';

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
    (view) => writeUniforms(view, inferredCount)
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
    (view) => writeUniforms(view, inferredCount),
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
    dispatch(device, pipeline, bindGroup, calcWorkgroups(opts), opName);
    uniformBuffer.destroy();
    const dtype = specDtype ? specDtype(opts, inputs) : (inputs[0] ? inputs[0].dtype : 'f32');
    return createTensor(outputBuf, dtype, outputShape(opts), `${opName}_output`);
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
    recordDispatch(recorder, pipeline, bindGroup, calcWorkgroups(opts), opName);
    const dtype = specDtype ? specDtype(opts, inputs) : (inputs[0] ? inputs[0].dtype : 'f32');
    return createTensor(outputBuf, dtype, outputShape(opts), `${opName}_output`);
  }

  return { run, record };
}
