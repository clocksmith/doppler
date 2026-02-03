

import { getDevice } from '../device.js';
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { getPipelineFast, createUniformBufferWithView } from './utils.js';
import { selectRuleValue } from './rule-registry.js';

function canUseF16(input) {
  return input.dtype === 'f16';
}


function selectSiLUVariant(isF16) {
  return selectRuleValue('silu', 'variant', { isF16 });
}


function resolveOverrides(group, context) {
  const overrides = selectRuleValue(group, 'overrides', context);
  return overrides && Object.keys(overrides).length > 0 ? overrides : null;
}


function selectSwiGLURowsplitBiasVariant(isF16) {
  return selectRuleValue('silu', 'swigluRowsplitBiasVariant', { isF16 });
}

function resolveSwigluLimit(value, context) {
  if (value === undefined) {
    throw new Error(`${context} requires an explicit swigluLimit (null or number).`);
  }
  if (value == null) return 0;
  return value;
}


function createSiLUBindGroupEntries(uniformBuffer, input, output, gate) {
  const gateBuffer = gate?.buffer ?? input.buffer;
  return [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: input.buffer } },
    { binding: 2, resource: { buffer: output } },
    { binding: 3, resource: { buffer: gateBuffer } },
  ];
}


export async function runSiLU(
  input,
  options = {}
) {
  const device = getDevice();
  const { size, gate = null, outputBuffer = null, useVec4 = false, swigluLimit } = options;
  const resolvedSwigluLimit = resolveSwigluLimit(swigluLimit, 'SiLU');

  const isF16 = canUseF16(input);
  const bytesPerElement = dtypeBytes(input.dtype);

  // Select variant using lookup table
  const variant = selectSiLUVariant(isF16);
  const overrides = resolveOverrides('silu', {
    hasGate: Boolean(gate),
    useVec4,
    useSplit: false,
    useRowsplit: false,
  });
  const pipeline = await getPipelineFast('silu', variant, null, overrides);

  const inferredSize = size || (input.buffer.size / bytesPerElement);
  const outputSize = inferredSize * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'silu_output');

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'silu_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredSize, true);
      view.setUint32(4, 0, true);
      view.setFloat32(8, gate ? resolvedSwigluLimit : 0, true);
      view.setFloat32(12, 0, true);
    },
    null,
    device
  );

  // Create bind group using helper
  const entries = createSiLUBindGroupEntries(uniformBuffer, input, output, gate);

  const bindGroup = device.createBindGroup({
    label: 'silu_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  const workgroups = Math.ceil(inferredSize / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'silu');

  uniformBuffer.destroy();

  return createTensor(output, input.dtype, [inferredSize], 'silu_output');
}


export async function runSwiGLURowsplitBias(
  input,
  bias,
  numTokens,
  dim,
  options = {}
) {
  const device = getDevice();
  const { outputBuffer = null, biasOffset = 0, swigluLimit } = options;
  const resolvedSwigluLimit = resolveSwigluLimit(swigluLimit, 'SwiGLU row-split');

  const useF16 = input.dtype === 'f16' && bias.dtype === 'f16';
  const variant = selectSwiGLURowsplitBiasVariant(useF16);
  const pipeline = await getPipelineFast('swiglu', variant);

  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = numTokens * dim * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'swiglu_output');

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'swiglu_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, dim, true);
      view.setUint32(8, biasOffset, true);
      view.setFloat32(12, resolvedSwigluLimit, true);
    },
    null,
    device
  );

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'swiglu_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: bias.buffer } },
      { binding: 3, resource: { buffer: output } },
    ],
  });

  const workgroups = Math.ceil((numTokens * dim) / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'swiglu');

  uniformBuffer.destroy();

  return createTensor(output, input.dtype, [numTokens, dim], 'swiglu_output');
}


export async function runSiLURowSplit(
  input,
  options
) {
  const device = getDevice();
  const { numTokens, dim, activation = 'silu', outputBuffer = null, swigluLimit } = options;
  const resolvedSwigluLimit = resolveSwigluLimit(swigluLimit, 'SiLU row-split');

  const isF16 = canUseF16(input);
  const bytesPerElement = dtypeBytes(input.dtype);

  const op = selectRuleValue('silu', 'activationOp', { activation });
  const variant = selectRuleValue(op, 'variant', { isF16 });
  const overrides = resolveOverrides(op, { useRowsplit: true });
  const pipeline = await getPipelineFast(op, variant, null, overrides);

  const outputSize = numTokens * dim * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'silu_rowsplit_output');

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'silu_rowsplit_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens * dim, true);  // size
      view.setUint32(4, dim, true);              // rowsplit_dim
      view.setFloat32(8, activation === 'silu' ? resolvedSwigluLimit : 0, true);
      view.setFloat32(12, 0, true);
    },
    null,
    device
  );

  // Bind group: provide a dummy gate buffer to satisfy the fixed layout
  const gateBuffer = input.buffer;
  const bindGroup = device.createBindGroup({
    label: 'silu_rowsplit_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: gateBuffer } },
    ],
  });

  const workgroups = Math.ceil((numTokens * dim) / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'silu_rowsplit');

  uniformBuffer.destroy();

  return createTensor(output, input.dtype, [numTokens, dim], 'silu_rowsplit_output');
}


export async function recordSiLURowSplit(
  recorder,
  input,
  options
) {
  const device = recorder.device;
  const { numTokens, dim, activation = 'silu', outputBuffer = null, swigluLimit } = options;
  const resolvedSwigluLimit = resolveSwigluLimit(swigluLimit, 'SiLU row-split');

  const isF16 = canUseF16(input);
  const bytesPerElement = dtypeBytes(input.dtype);

  const op = selectRuleValue('silu', 'activationOp', { activation });
  const variant = selectRuleValue(op, 'variant', { isF16 });
  const overrides = resolveOverrides(op, { useRowsplit: true });
  const pipeline = await getPipelineFast(op, variant, null, overrides);

  const outputSize = numTokens * dim * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'silu_rowsplit_output');

  // Uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'silu_rowsplit_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens * dim, true);  // size
      view.setUint32(4, dim, true);              // rowsplit_dim
      view.setFloat32(8, activation === 'silu' ? resolvedSwigluLimit : 0, true);
      view.setFloat32(12, 0, true);
    },
    recorder
  );

  const gateBuffer = input.buffer;
  const bindGroup = device.createBindGroup({
    label: 'silu_rowsplit_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: gateBuffer } },
    ],
  });

  const workgroups = Math.ceil((numTokens * dim) / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'silu_rowsplit');

  return createTensor(output, input.dtype, [numTokens, dim], 'silu_rowsplit_output');
}


export async function recordSiLU(
  recorder,
  input,
  options = {}
) {
  const device = recorder.device;
  const { size, gate = null, outputBuffer = null, swigluLimit } = options;
  const resolvedSwigluLimit = resolveSwigluLimit(swigluLimit, 'SiLU');

  const isF16 = canUseF16(input);
  const bytesPerElement = dtypeBytes(input.dtype);

  // Select variant using lookup table
  const variant = selectSiLUVariant(isF16);
  const overrides = resolveOverrides('silu', {
    hasGate: Boolean(gate),
    useVec4: false,
    useSplit: false,
    useRowsplit: false,
  });
  const pipeline = await getPipelineFast('silu', variant, null, overrides);

  const inferredSize = size || (input.buffer.size / bytesPerElement);
  const outputSize = inferredSize * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'silu_output');

  // Uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'silu_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredSize, true);
      view.setUint32(4, 0, true);
      view.setFloat32(8, gate ? resolvedSwigluLimit : 0, true);
      view.setFloat32(12, 0, true);
    },
    recorder
  );

  // Create bind group using helper
  const entries = createSiLUBindGroupEntries(uniformBuffer, input, output, gate);

  const bindGroup = device.createBindGroup({
    label: 'silu_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  const workgroups = Math.ceil(inferredSize / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'silu');

  return createTensor(output, input.dtype, [inferredSize], 'silu_output');
}
