

import { getDevice } from '../device.js';
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView } from './utils.js';
import { selectRuleValue } from './rule-registry.js';


export async function runScale(
  input,
  scale,
  options = {}
) {
  const device = getDevice();
  const { count, outputBuffer = null, inplace = false } = options;

  const bytesPerElement = dtypeBytes(input.dtype);
  const inferredCount = count ?? Math.floor(input.buffer.size / bytesPerElement);
  const variant = selectRuleValue('scale', 'variant', { inplace });
  const pipeline = await createPipeline('scale', variant);

  const outputSize = inferredCount * bytesPerElement;
  const outputBuf = inplace ? input.buffer : (outputBuffer || acquireBuffer(outputSize, undefined, 'scale_output'));

  // Create uniform buffer (16 bytes to match WGSL struct with padding)
  const uniformBuffer = createUniformBufferWithView(
    'scale_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredCount, true);
      view.setFloat32(4, scale, true);
      // _pad0 and _pad1 at offsets 8 and 12 (unused)
    },
    null,
    device
  );

  const bindGroupEntries = inplace
    ? [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 2, resource: { buffer: outputBuf } },
    ]
    : [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: outputBuf } },
    ];

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'scale_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: bindGroupEntries,
  });

  const workgroups = Math.ceil(inferredCount / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'scale');

  uniformBuffer.destroy();

  return createTensor(outputBuf, input.dtype, [...input.shape], 'scale_output');
}


export async function recordScale(
  recorder,
  input,
  scale,
  options = {}
) {
  const device = recorder.device;
  const { count, outputBuffer = null, inplace = false } = options;

  const bytesPerElement = dtypeBytes(input.dtype);
  const inferredCount = count ?? Math.floor(input.buffer.size / bytesPerElement);
  const variant = selectRuleValue('scale', 'variant', { inplace });
  const pipeline = await createPipeline('scale', variant);

  const outputSize = inferredCount * bytesPerElement;
  const outputBuf = inplace ? input.buffer : (outputBuffer || acquireBuffer(outputSize, undefined, 'scale_output'));

  // Create uniform buffer via recorder (tracked for cleanup, 16 bytes to match WGSL)
  const uniformBuffer = createUniformBufferWithView(
    'scale_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredCount, true);
      view.setFloat32(4, scale, true);
      // _pad0 and _pad1 at offsets 8 and 12 (unused)
    },
    recorder
  );

  const bindGroupEntries = inplace
    ? [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 2, resource: { buffer: outputBuf } },
    ]
    : [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: outputBuf } },
    ];

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'scale_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: bindGroupEntries,
  });

  const workgroups = Math.ceil(inferredCount / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'scale');

  return createTensor(outputBuf, input.dtype, [...input.shape], 'scale_output');
}
