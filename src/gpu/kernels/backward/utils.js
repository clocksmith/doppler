import { getDevice } from '../../device.js';
import { acquireBuffer } from '../../buffer-pool.js';
import { createTensor, dtypeBytes } from '../../tensor.js';
import { WORKGROUP_SIZES } from '../constants.js';
import { dispatch, recordDispatch } from '../dispatch.js';
import { createPipeline, createUniformBufferWithView } from '../utils.js';

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
    },
    null,
    device
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

  uniformBuffer.destroy();

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
