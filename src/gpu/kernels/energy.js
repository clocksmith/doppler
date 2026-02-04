import { getDevice } from '../device.js';
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { getPipelineFast, createUniformBufferWithView } from './utils.js';
import { selectRuleValue } from './rule-registry.js';

function ensureMatchingDtype(state, target, op) {
  if (state.dtype !== target.dtype) {
    throw new Error(`${op}: state dtype ${state.dtype} does not match target dtype ${target.dtype}.`);
  }
  if (state.dtype !== 'f16' && state.dtype !== 'f32') {
    throw new Error(`${op}: unsupported dtype ${state.dtype}.`);
  }
}

function inferCount(tensor, countOverride) {
  if (Number.isFinite(countOverride) && countOverride > 0) {
    return Math.floor(countOverride);
  }
  if (Array.isArray(tensor.shape) && tensor.shape.length > 0) {
    return tensor.shape.reduce((acc, value) => acc * value, 1);
  }
  return Math.floor(tensor.buffer.size / dtypeBytes(tensor.dtype));
}

function selectEnergyEvalVariant(dtype) {
  return selectRuleValue('energy', 'evalVariant', { isF16: dtype === 'f16' });
}

function selectEnergyUpdateVariant(dtype) {
  return selectRuleValue('energy', 'updateVariant', { isF16: dtype === 'f16' });
}

function selectEnergyQuintelUpdateVariant(dtype) {
  return selectRuleValue('energy', 'quintelUpdateVariant', { isF16: dtype === 'f16' });
}

function selectEnergyQuintelReduceVariant(dtype) {
  return selectRuleValue('energy', 'quintelReduceVariant', { isF16: dtype === 'f16' });
}

function resolveQuintelSize(state, sizeOverride) {
  if (Number.isFinite(sizeOverride) && sizeOverride > 0) {
    return Math.floor(sizeOverride);
  }
  if (Array.isArray(state.shape) && state.shape.length >= 2) {
    return Math.max(1, Math.floor(state.shape[0]));
  }
  return null;
}

function buildQuintelFlags(rules, binarizeWeight) {
  let flags = 0;
  if (rules?.mirrorX) flags |= 1;
  if (rules?.mirrorY) flags |= 2;
  if (rules?.diagonal) flags |= 4;
  if (rules?.count) flags |= 8;
  if (rules?.center) flags |= 16;
  if (Number.isFinite(binarizeWeight) && binarizeWeight !== 0) flags |= 32;
  return flags >>> 0;
}

export async function runEnergyEval(
  state,
  target,
  options = {}
) {
  ensureMatchingDtype(state, target, 'runEnergyEval');
  const device = getDevice();
  const { count, scale = 1.0, outputBuffer = null } = options;
  const elementCount = inferCount(state, count);

  const outputSize = elementCount * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'energy_eval_output');

  const variant = selectEnergyEvalVariant(state.dtype);
  const pipeline = await getPipelineFast('energy_eval', variant);

  const uniformBuffer = createUniformBufferWithView(
    'energy_eval_uniforms',
    16,
    (view) => {
      view.setUint32(0, elementCount, true);
      view.setFloat32(4, scale, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'energy_eval_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: state.buffer } },
      { binding: 2, resource: { buffer: target.buffer } },
      { binding: 3, resource: { buffer: output } },
    ],
  });

  const workgroups = Math.ceil(elementCount / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'energy_eval');

  uniformBuffer.destroy();

  return createTensor(output, 'f32', [elementCount], 'energy_eval_output');
}

export async function recordEnergyEval(
  recorder,
  state,
  target,
  options = {}
) {
  ensureMatchingDtype(state, target, 'recordEnergyEval');
  const device = recorder.device;
  const { count, scale = 1.0, outputBuffer = null } = options;
  const elementCount = inferCount(state, count);

  const outputSize = elementCount * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'energy_eval_output');

  const variant = selectEnergyEvalVariant(state.dtype);
  const pipeline = await getPipelineFast('energy_eval', variant);

  const uniformBuffer = createUniformBufferWithView(
    'energy_eval_uniforms',
    16,
    (view) => {
      view.setUint32(0, elementCount, true);
      view.setFloat32(4, scale, true);
    },
    recorder
  );

  const bindGroup = device.createBindGroup({
    label: 'energy_eval_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: state.buffer } },
      { binding: 2, resource: { buffer: target.buffer } },
      { binding: 3, resource: { buffer: output } },
    ],
  });

  const workgroups = Math.ceil(elementCount / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'energy_eval');

  return createTensor(output, 'f32', [elementCount], 'energy_eval_output');
}

export async function runEnergyUpdate(
  state,
  target,
  options = {}
) {
  ensureMatchingDtype(state, target, 'runEnergyUpdate');
  const device = getDevice();
  const { count, stepSize = 0.1, gradientScale = 1.0 } = options;
  const elementCount = inferCount(state, count);

  const variant = selectEnergyUpdateVariant(state.dtype);
  const pipeline = await getPipelineFast('energy_update', variant);

  const uniformBuffer = createUniformBufferWithView(
    'energy_update_uniforms',
    16,
    (view) => {
      view.setUint32(0, elementCount, true);
      view.setFloat32(4, stepSize, true);
      view.setFloat32(8, gradientScale, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'energy_update_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: state.buffer } },
      { binding: 2, resource: { buffer: target.buffer } },
    ],
  });

  const workgroups = Math.ceil(elementCount / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'energy_update');

  uniformBuffer.destroy();

  return state;
}

export async function recordEnergyUpdate(
  recorder,
  state,
  target,
  options = {}
) {
  ensureMatchingDtype(state, target, 'recordEnergyUpdate');
  const device = recorder.device;
  const { count, stepSize = 0.1, gradientScale = 1.0 } = options;
  const elementCount = inferCount(state, count);

  const variant = selectEnergyUpdateVariant(state.dtype);
  const pipeline = await getPipelineFast('energy_update', variant);

  const uniformBuffer = createUniformBufferWithView(
    'energy_update_uniforms',
    16,
    (view) => {
      view.setUint32(0, elementCount, true);
      view.setFloat32(4, stepSize, true);
      view.setFloat32(8, gradientScale, true);
    },
    recorder
  );

  const bindGroup = device.createBindGroup({
    label: 'energy_update_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: state.buffer } },
      { binding: 2, resource: { buffer: target.buffer } },
    ],
  });

  const workgroups = Math.ceil(elementCount / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'energy_update');

  return state;
}

export async function runEnergyQuintelUpdate(
  state,
  options = {}
) {
  if (state.dtype !== 'f16' && state.dtype !== 'f32') {
    throw new Error(`runEnergyQuintelUpdate: unsupported dtype ${state.dtype}.`);
  }
  const device = getDevice();
  const {
    count,
    size,
    stepSize = 0.1,
    gradientScale = 1.0,
    countDiff = 0.0,
    symmetryWeight = 1.0,
    countWeight = 1.0,
    centerWeight = 1.0,
    binarizeWeight = 0.0,
    centerTarget = 1.0,
    clampMin = 0.0,
    clampMax = 1.0,
    rules = {},
  } = options;
  const elementCount = inferCount(state, count);
  const boardSize = resolveQuintelSize(state, size);
  if (!boardSize) {
    throw new Error('runEnergyQuintelUpdate: size is required for quintel update.');
  }

  const variant = selectEnergyQuintelUpdateVariant(state.dtype);
  const pipeline = await getPipelineFast('energy_quintel_update', variant);
  const flags = buildQuintelFlags(rules, binarizeWeight);

  const uniformBuffer = createUniformBufferWithView(
    'energy_quintel_uniforms',
    64,
    (view) => {
      view.setUint32(0, elementCount, true);
      view.setUint32(4, boardSize, true);
      view.setUint32(8, flags, true);
      view.setFloat32(16, stepSize, true);
      view.setFloat32(20, gradientScale, true);
      view.setFloat32(24, countDiff, true);
      view.setFloat32(28, centerTarget, true);
      view.setFloat32(32, symmetryWeight, true);
      view.setFloat32(36, countWeight, true);
      view.setFloat32(40, centerWeight, true);
      view.setFloat32(44, binarizeWeight, true);
      view.setFloat32(48, clampMin, true);
      view.setFloat32(52, clampMax, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'energy_quintel_update_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: state.buffer } },
    ],
  });

  const workgroups = Math.ceil(elementCount / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'energy_quintel_update');

  uniformBuffer.destroy();

  return state;
}

export async function runEnergyQuintelReduce(
  state,
  options = {}
) {
  if (state.dtype !== 'f16' && state.dtype !== 'f32') {
    throw new Error(`runEnergyQuintelReduce: unsupported dtype ${state.dtype}.`);
  }
  const device = getDevice();
  const {
    count,
    size,
    symmetryWeight = 1.0,
    centerWeight = 1.0,
    binarizeWeight = 0.0,
    centerTarget = 1.0,
    rules = {},
    outputBuffer = null,
  } = options;
  const elementCount = inferCount(state, count);
  const boardSize = resolveQuintelSize(state, size);
  if (!boardSize) {
    throw new Error('runEnergyQuintelReduce: size is required for quintel reduction.');
  }

  const variant = selectEnergyQuintelReduceVariant(state.dtype);
  const pipeline = await getPipelineFast('energy_quintel_reduce', variant);
  const flags = buildQuintelFlags(rules, binarizeWeight);

  const uniformBuffer = createUniformBufferWithView(
    'energy_quintel_reduce_uniforms',
    48,
    (view) => {
      view.setUint32(0, elementCount, true);
      view.setUint32(4, boardSize, true);
      view.setUint32(8, flags, true);
      view.setFloat32(16, symmetryWeight, true);
      view.setFloat32(20, centerWeight, true);
      view.setFloat32(24, binarizeWeight, true);
      view.setFloat32(28, centerTarget, true);
    },
    null,
    device
  );

  const workgroups = Math.ceil(elementCount / WORKGROUP_SIZES.DEFAULT);
  const outputSize = workgroups * 16;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'energy_quintel_reduce_output');

  const bindGroup = device.createBindGroup({
    label: 'energy_quintel_reduce_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: state.buffer } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  dispatch(device, pipeline, bindGroup, workgroups, 'energy_quintel_reduce');

  uniformBuffer.destroy();

  return createTensor(output, 'f32', [workgroups, 4], 'energy_quintel_reduce_output');
}

export async function recordEnergyQuintelUpdate(
  recorder,
  state,
  options = {}
) {
  if (state.dtype !== 'f16' && state.dtype !== 'f32') {
    throw new Error(`recordEnergyQuintelUpdate: unsupported dtype ${state.dtype}.`);
  }
  const device = recorder.device;
  const {
    count,
    size,
    stepSize = 0.1,
    gradientScale = 1.0,
    countDiff = 0.0,
    symmetryWeight = 1.0,
    countWeight = 1.0,
    centerWeight = 1.0,
    binarizeWeight = 0.0,
    centerTarget = 1.0,
    clampMin = 0.0,
    clampMax = 1.0,
    rules = {},
  } = options;
  const elementCount = inferCount(state, count);
  const boardSize = resolveQuintelSize(state, size);
  if (!boardSize) {
    throw new Error('recordEnergyQuintelUpdate: size is required for quintel update.');
  }

  const variant = selectEnergyQuintelUpdateVariant(state.dtype);
  const pipeline = await getPipelineFast('energy_quintel_update', variant);
  const flags = buildQuintelFlags(rules, binarizeWeight);

  const uniformBuffer = createUniformBufferWithView(
    'energy_quintel_uniforms',
    64,
    (view) => {
      view.setUint32(0, elementCount, true);
      view.setUint32(4, boardSize, true);
      view.setUint32(8, flags, true);
      view.setFloat32(16, stepSize, true);
      view.setFloat32(20, gradientScale, true);
      view.setFloat32(24, countDiff, true);
      view.setFloat32(28, centerTarget, true);
      view.setFloat32(32, symmetryWeight, true);
      view.setFloat32(36, countWeight, true);
      view.setFloat32(40, centerWeight, true);
      view.setFloat32(44, binarizeWeight, true);
      view.setFloat32(48, clampMin, true);
      view.setFloat32(52, clampMax, true);
    },
    recorder
  );

  const bindGroup = device.createBindGroup({
    label: 'energy_quintel_update_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: state.buffer } },
    ],
  });

  const workgroups = Math.ceil(elementCount / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'energy_quintel_update');

  return state;
}

export async function recordEnergyQuintelReduce(
  recorder,
  state,
  options = {}
) {
  if (state.dtype !== 'f16' && state.dtype !== 'f32') {
    throw new Error(`recordEnergyQuintelReduce: unsupported dtype ${state.dtype}.`);
  }
  const device = recorder.device;
  const {
    count,
    size,
    symmetryWeight = 1.0,
    centerWeight = 1.0,
    binarizeWeight = 0.0,
    centerTarget = 1.0,
    rules = {},
    outputBuffer = null,
  } = options;
  const elementCount = inferCount(state, count);
  const boardSize = resolveQuintelSize(state, size);
  if (!boardSize) {
    throw new Error('recordEnergyQuintelReduce: size is required for quintel reduction.');
  }

  const variant = selectEnergyQuintelReduceVariant(state.dtype);
  const pipeline = await getPipelineFast('energy_quintel_reduce', variant);
  const flags = buildQuintelFlags(rules, binarizeWeight);

  const uniformBuffer = createUniformBufferWithView(
    'energy_quintel_reduce_uniforms',
    48,
    (view) => {
      view.setUint32(0, elementCount, true);
      view.setUint32(4, boardSize, true);
      view.setUint32(8, flags, true);
      view.setFloat32(16, symmetryWeight, true);
      view.setFloat32(20, centerWeight, true);
      view.setFloat32(24, binarizeWeight, true);
      view.setFloat32(28, centerTarget, true);
    },
    recorder
  );

  const workgroups = Math.ceil(elementCount / WORKGROUP_SIZES.DEFAULT);
  const outputSize = workgroups * 16;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'energy_quintel_reduce_output');

  const bindGroup = device.createBindGroup({
    label: 'energy_quintel_reduce_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: state.buffer } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'energy_quintel_reduce');

  return createTensor(output, 'f32', [workgroups, 4], 'energy_quintel_reduce_output');
}
