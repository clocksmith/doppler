

import { getDevice } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../buffer-pool.js';
import { createTensor, inferOutputDtype, dtypeBytes } from '../tensor.js';
import { WORKGROUP_SIZES, VEC4_ELEMENTS_PER_WG } from './constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { getPipelineFast, createUniformBufferWithView } from './utils.js';
import { castF16ToF32, castF32ToF16, recordCastF16ToF32, recordCastF32ToF16 } from './cast.js';
import { selectRuleValue } from './rule-registry.js';


function selectResidualVariant(outputDtype, useVec4) {
  return selectRuleValue(
    'residual',
    'residualVariant',
    { outputDtype, useVec4 }
  );
}


function selectBiasAddVariant(dataDtype, biasDtype) {
  return selectRuleValue(
    'residual',
    'biasAddVariant',
    { dataDtype, biasDtype }
  );
}


async function alignResidualInputs(
  a,
  b,
  recorder
) {
  if (a.dtype === b.dtype) {
    return { a, b, temps: [] };
  }

  if (a.dtype === 'f16' && b.dtype === 'f32') {
    const casted = recorder ? await recordCastF16ToF32(recorder, a) : await castF16ToF32(a);
    return { a: casted, b, temps: [casted.buffer] };
  }

  if (a.dtype === 'f32' && b.dtype === 'f16') {
    const casted = recorder ? await recordCastF16ToF32(recorder, b) : await castF16ToF32(b);
    return { a, b: casted, temps: [casted.buffer] };
  }

  return { a, b, temps: [] };
}


async function alignBiasTensor(
  data,
  bias,
  recorder
) {
  if (data.dtype === bias.dtype) {
    return { bias, temps: [] };
  }

  if (data.dtype === 'f16' && bias.dtype === 'f32') {
    const casted = recorder ? await recordCastF32ToF16(recorder, bias) : await castF32ToF16(bias);
    return { bias: casted, temps: [casted.buffer] };
  }

  if (data.dtype === 'f32' && bias.dtype === 'f16') {
    const casted = recorder ? await recordCastF16ToF32(recorder, bias) : await castF16ToF32(bias);
    return { bias: casted, temps: [casted.buffer] };
  }

  return { bias, temps: [] };
}


export async function runResidualAdd(
  a,
  b,
  size,
  options = {}
) {
  const device = getDevice();
  const { useVec4 = true, outputBuffer = null } = options;

  const { a: aAligned, b: bAligned, temps } = await alignResidualInputs(a, b);
  const outputDtype = inferOutputDtype(aAligned, bAligned);
  const bytesPerElement = dtypeBytes(outputDtype);

  const variant = selectResidualVariant(outputDtype, useVec4);
  const pipeline = await getPipelineFast('residual', variant);

  const outputSize = size * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'residual_output');

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'residual_uniforms',
    16,
    (view) => {
      view.setUint32(0, size, true);
    },
    null,
    device
  );

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'residual_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: aAligned.buffer } },
      { binding: 2, resource: { buffer: bAligned.buffer } },
      { binding: 3, resource: { buffer: output } },
    ],
  });

  // residual.wgsl: main uses @workgroup_size(256), add_vec4 uses @workgroup_size(64)
  // vec4 variant: 64 threads * 4 elements = 256 elements per workgroup
  const workgroups = useVec4
    ? Math.ceil(size / VEC4_ELEMENTS_PER_WG)
    : Math.ceil(size / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'residual');

  uniformBuffer.destroy();

  for (const temp of temps) {
    releaseBuffer(temp);
  }

  return createTensor(output, outputDtype, [size], 'residual_output');
}


export async function runBiasAdd(
  data,
  bias,
  numTokens,
  dim,
  options = {}
) {
  const device = getDevice();
  const { dataOffset = 0, biasOffset = 0 } = options;

  const { bias: biasAligned, temps } = await alignBiasTensor(data, bias);
  const variant = selectBiasAddVariant(data.dtype, biasAligned.dtype);
  const pipeline = await getPipelineFast('bias_add', variant);

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'bias_add_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, dim, true);
      view.setUint32(8, dataOffset, true);
      view.setUint32(12, biasOffset, true);
    },
    null,
    device
  );

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'bias_add_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: data.buffer } },
      { binding: 2, resource: { buffer: biasAligned.buffer } },
    ],
  });

  const workgroups = Math.ceil((numTokens * dim) / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'bias_add');

  uniformBuffer.destroy();

  for (const temp of temps) {
    releaseBuffer(temp);
  }

  // Bias add is in-place, return tensor with same buffer
  return createTensor(data.buffer, data.dtype, [numTokens, dim], 'bias_add_output');
}


export async function recordResidualAdd(
  recorder,
  a,
  b,
  size,
  options = {}
) {
  const device = recorder.device;
  const { outputBuffer = null, useVec4 = true } = options;

  const { a: aAligned, b: bAligned, temps } = await alignResidualInputs(a, b, recorder);
  const outputDtype = inferOutputDtype(aAligned, bAligned);
  const bytesPerElement = dtypeBytes(outputDtype);

  const variant = selectResidualVariant(outputDtype, useVec4);
  const pipeline = await getPipelineFast('residual', variant);

  const outputSize = size * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'residual_output');

  // Uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'residual_uniforms',
    16,
    (view) => {
      view.setUint32(0, size, true);
    },
    recorder
  );

  // Bind group
  const bindGroup = device.createBindGroup({
    label: 'residual_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: aAligned.buffer } },
      { binding: 2, resource: { buffer: bAligned.buffer } },
      { binding: 3, resource: { buffer: output } },
    ],
  });

  const workgroups = useVec4
    ? Math.ceil(size / VEC4_ELEMENTS_PER_WG)
    : Math.ceil(size / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'residual');

  for (const temp of temps) {
    recorder.trackTemporaryBuffer(temp);
  }

  return createTensor(output, outputDtype, [size], 'residual_output');
}


export async function recordBiasAdd(
  recorder,
  data,
  bias,
  numTokens,
  dim,
  options = {}
) {
  const device = recorder.device;
  const { dataOffset = 0, biasOffset = 0 } = options;

  const { bias: biasAligned, temps } = await alignBiasTensor(data, bias, recorder);
  const variant = selectBiasAddVariant(data.dtype, biasAligned.dtype);
  const pipeline = await getPipelineFast('bias_add', variant);

  // Uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'bias_add_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, dim, true);
      view.setUint32(8, dataOffset, true);
      view.setUint32(12, biasOffset, true);
    },
    recorder
  );

  // Bind group
  const bindGroup = device.createBindGroup({
    label: 'bias_add_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: data.buffer } },
      { binding: 2, resource: { buffer: biasAligned.buffer } },
    ],
  });

  const workgroups = Math.ceil((numTokens * dim) / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'bias_add');

  for (const temp of temps) {
    recorder.trackTemporaryBuffer(temp);
  }

  // Bias add is in-place, return tensor with same buffer
  return createTensor(data.buffer, data.dtype, [numTokens, dim], 'bias_add_output');
}
