

import { getDevice } from '../device.js';
import { acquireBuffer } from '../buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView } from './utils.js';


function canUseF16(input) {
  return input.dtype === 'f16';
}


export async function runGeLU(
  input,
  options = {}
) {
  const device = getDevice();
  const { size, gate = null, outputBuffer = null } = options;

  const isF16 = canUseF16(input);
  const bytesPerElement = dtypeBytes(input.dtype);

  // Select gated variant when gate buffer is provided
  
  let variant;
  if (gate) {
    variant = isF16 ? 'geglu_f16' : 'geglu';
  } else {
    variant = isF16 ? 'gelu_f16' : 'gelu';
  }
  const pipeline = await createPipeline('gelu', variant);

  const inferredSize = size || (input.buffer.size / bytesPerElement);
  const outputSize = inferredSize * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'gelu_output');

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'gelu_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredSize, true);
      view.setUint32(4, 0, true);
    },
    null,
    device
  );

  // Create bind group
  
  const entries = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: input.buffer } },
    { binding: 2, resource: { buffer: output } },
  ];
  if (gate) {
    entries.push({ binding: 3, resource: { buffer: gate.buffer } });
  }
  const bindGroup = device.createBindGroup({
    label: 'gelu_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  const workgroups = Math.ceil(inferredSize / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'gelu');

  uniformBuffer.destroy();

  return createTensor(output, input.dtype, [inferredSize], 'gelu_output');
}


export async function recordGeLU(
  recorder,
  input,
  options = {}
) {
  const device = recorder.device;
  const { size, gate = null, outputBuffer = null } = options;

  const isF16 = canUseF16(input);
  const bytesPerElement = dtypeBytes(input.dtype);

  // Select gated variant when gate buffer is provided
  
  let variant;
  if (gate) {
    variant = isF16 ? 'geglu_f16' : 'geglu';
  } else {
    variant = isF16 ? 'gelu_f16' : 'gelu';
  }
  const pipeline = await createPipeline('gelu', variant);

  const inferredSize = size || (input.buffer.size / bytesPerElement);
  const outputSize = inferredSize * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'gelu_output');

  // Uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'gelu_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredSize, true);
      view.setUint32(4, 0, true);
    },
    recorder
  );

  // Bind group entries - gate variant needs binding 3
  
  const entries = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: input.buffer } },
    { binding: 2, resource: { buffer: output } },
  ];

  // Add gate binding for gated variant
  if (gate) {
    entries.push({ binding: 3, resource: { buffer: gate.buffer } });
  }

  const bindGroup = device.createBindGroup({
    label: 'gelu_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  const workgroups = Math.ceil(inferredSize / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'gelu');

  return createTensor(output, input.dtype, [inferredSize], 'gelu_output');
}
