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

function ensureSupportedEnergyDtype(state, op) {
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

function selectEnergyQuintelGradVariant(dtype) {
  return selectRuleValue('energy', 'quintelGradVariant', { isF16: dtype === 'f16' });
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

function resolveExecution(recorder) {
  return {
    recorder: recorder || null,
    device: recorder?.device || getDevice(),
  };
}

function createUniformBuffer(execution, label, size, writeUniforms) {
  return createUniformBufferWithView(
    label,
    size,
    writeUniforms,
    execution.recorder,
    execution.device
  );
}

function dispatchEnergy(execution, pipeline, bindGroup, workgroups, label) {
  if (execution.recorder) {
    recordDispatch(execution.recorder, pipeline, bindGroup, workgroups, label);
    return;
  }
  dispatch(execution.device, pipeline, bindGroup, workgroups, label);
}

function releaseUniformBuffer(execution, uniformBuffer) {
  if (!execution.recorder) {
    uniformBuffer.destroy();
  }
}

function writeQuintelUpdateUniform(view, params) {
  view.setUint32(0, params.elementCount, true);
  view.setUint32(4, params.boardSize, true);
  view.setUint32(8, params.flags, true);
  view.setFloat32(16, params.stepSize, true);
  view.setFloat32(20, params.gradientScale, true);
  view.setFloat32(24, params.countDiff, true);
  view.setFloat32(28, params.centerTarget, true);
  view.setFloat32(32, params.symmetryWeight, true);
  view.setFloat32(36, params.countWeight, true);
  view.setFloat32(40, params.centerWeight, true);
  view.setFloat32(44, params.binarizeWeight, true);
  view.setFloat32(48, params.clampMin, true);
  view.setFloat32(52, params.clampMax, true);
}

function writeQuintelReduceUniform(view, params) {
  view.setUint32(0, params.elementCount, true);
  view.setUint32(4, params.boardSize, true);
  view.setUint32(8, params.flags, true);
  view.setFloat32(16, params.symmetryWeight, true);
  view.setFloat32(20, params.centerWeight, true);
  view.setFloat32(24, params.binarizeWeight, true);
  view.setFloat32(28, params.centerTarget, true);
}

function writeQuintelGradUniform(view, params) {
  view.setUint32(0, params.elementCount, true);
  view.setUint32(4, params.boardSize, true);
  view.setUint32(8, params.flags, true);
  view.setFloat32(24, params.countDiff, true);
  view.setFloat32(28, params.centerTarget, true);
  view.setFloat32(32, params.symmetryWeight, true);
  view.setFloat32(36, params.countWeight, true);
  view.setFloat32(40, params.centerWeight, true);
  view.setFloat32(44, params.binarizeWeight, true);
}

async function executeEnergyEval(recorder, state, target, options = {}, op) {
  ensureMatchingDtype(state, target, op);
  const execution = resolveExecution(recorder);
  const { count, scale = 1.0, outputBuffer = null } = options;
  const elementCount = inferCount(state, count);

  const outputSize = elementCount * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'energy_eval_output');

  const variant = selectEnergyEvalVariant(state.dtype);
  const pipeline = await getPipelineFast('energy_eval', variant);

  const uniformBuffer = createUniformBuffer(execution, 'energy_eval_uniforms', 16, (view) => {
    view.setUint32(0, elementCount, true);
    view.setFloat32(4, scale, true);
  });

  const bindGroup = execution.device.createBindGroup({
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
  dispatchEnergy(execution, pipeline, bindGroup, workgroups, 'energy_eval');
  releaseUniformBuffer(execution, uniformBuffer);

  return createTensor(output, 'f32', [elementCount], 'energy_eval_output');
}

async function executeEnergyUpdate(recorder, state, target, options = {}, op) {
  ensureMatchingDtype(state, target, op);
  const execution = resolveExecution(recorder);
  const { count, stepSize = 0.1, gradientScale = 1.0 } = options;
  const elementCount = inferCount(state, count);

  const variant = selectEnergyUpdateVariant(state.dtype);
  const pipeline = await getPipelineFast('energy_update', variant);

  const uniformBuffer = createUniformBuffer(execution, 'energy_update_uniforms', 16, (view) => {
    view.setUint32(0, elementCount, true);
    view.setFloat32(4, stepSize, true);
    view.setFloat32(8, gradientScale, true);
  });

  const bindGroup = execution.device.createBindGroup({
    label: 'energy_update_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: state.buffer } },
      { binding: 2, resource: { buffer: target.buffer } },
    ],
  });

  const workgroups = Math.ceil(elementCount / WORKGROUP_SIZES.DEFAULT);
  dispatchEnergy(execution, pipeline, bindGroup, workgroups, 'energy_update');
  releaseUniformBuffer(execution, uniformBuffer);

  return state;
}

async function executeEnergyQuintelUpdate(recorder, state, options = {}, op) {
  ensureSupportedEnergyDtype(state, op);
  const execution = resolveExecution(recorder);
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
    throw new Error(`${op}: size is required for quintel update.`);
  }

  const variant = selectEnergyQuintelUpdateVariant(state.dtype);
  const pipeline = await getPipelineFast('energy_quintel_update', variant);
  const flags = buildQuintelFlags(rules, binarizeWeight);

  const uniformBuffer = createUniformBuffer(execution, 'energy_quintel_uniforms', 64, (view) => {
    writeQuintelUpdateUniform(view, {
      elementCount,
      boardSize,
      flags,
      stepSize,
      gradientScale,
      countDiff,
      symmetryWeight,
      countWeight,
      centerWeight,
      binarizeWeight,
      centerTarget,
      clampMin,
      clampMax,
    });
  });

  const bindGroup = execution.device.createBindGroup({
    label: 'energy_quintel_update_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: state.buffer } },
    ],
  });

  const workgroups = Math.ceil(elementCount / WORKGROUP_SIZES.DEFAULT);
  dispatchEnergy(execution, pipeline, bindGroup, workgroups, 'energy_quintel_update');
  releaseUniformBuffer(execution, uniformBuffer);

  return state;
}

async function executeEnergyQuintelReduce(recorder, state, options = {}, op) {
  ensureSupportedEnergyDtype(state, op);
  const execution = resolveExecution(recorder);
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
    throw new Error(`${op}: size is required for quintel reduction.`);
  }

  const variant = selectEnergyQuintelReduceVariant(state.dtype);
  const pipeline = await getPipelineFast('energy_quintel_reduce', variant);
  const flags = buildQuintelFlags(rules, binarizeWeight);

  const uniformBuffer = createUniformBuffer(execution, 'energy_quintel_reduce_uniforms', 48, (view) => {
    writeQuintelReduceUniform(view, {
      elementCount,
      boardSize,
      flags,
      symmetryWeight,
      centerWeight,
      binarizeWeight,
      centerTarget,
    });
  });

  const workgroups = Math.ceil(elementCount / WORKGROUP_SIZES.DEFAULT);
  const outputSize = workgroups * 16;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'energy_quintel_reduce_output');

  const bindGroup = execution.device.createBindGroup({
    label: 'energy_quintel_reduce_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: state.buffer } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  dispatchEnergy(execution, pipeline, bindGroup, workgroups, 'energy_quintel_reduce');
  releaseUniformBuffer(execution, uniformBuffer);

  return createTensor(output, 'f32', [workgroups, 4], 'energy_quintel_reduce_output');
}

async function executeEnergyQuintelGrad(recorder, state, options = {}, op) {
  ensureSupportedEnergyDtype(state, op);
  const execution = resolveExecution(recorder);
  const {
    count,
    size,
    countDiff = 0.0,
    symmetryWeight = 1.0,
    countWeight = 1.0,
    centerWeight = 1.0,
    binarizeWeight = 0.0,
    centerTarget = 1.0,
    rules = {},
    outputBuffer = null,
  } = options;
  const elementCount = inferCount(state, count);
  const boardSize = resolveQuintelSize(state, size);
  if (!boardSize) {
    throw new Error(`${op}: size is required for quintel gradient.`);
  }

  const variant = selectEnergyQuintelGradVariant(state.dtype);
  const pipeline = await getPipelineFast('energy_quintel_grad', variant);
  const flags = buildQuintelFlags(rules, binarizeWeight);

  const uniformBuffer = createUniformBuffer(execution, 'energy_quintel_grad_uniforms', 64, (view) => {
    writeQuintelGradUniform(view, {
      elementCount,
      boardSize,
      flags,
      countDiff,
      symmetryWeight,
      countWeight,
      centerWeight,
      binarizeWeight,
      centerTarget,
    });
  });

  const outputSize = elementCount * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'energy_quintel_grad_output');

  const bindGroup = execution.device.createBindGroup({
    label: 'energy_quintel_grad_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: state.buffer } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  const workgroups = Math.ceil(elementCount / WORKGROUP_SIZES.DEFAULT);
  dispatchEnergy(execution, pipeline, bindGroup, workgroups, 'energy_quintel_grad');
  releaseUniformBuffer(execution, uniformBuffer);

  return createTensor(output, 'f32', [elementCount], 'energy_quintel_grad_output');
}

export async function runEnergyEval(state, target, options = {}) {
  return executeEnergyEval(null, state, target, options, 'runEnergyEval');
}

export async function recordEnergyEval(recorder, state, target, options = {}) {
  return executeEnergyEval(recorder, state, target, options, 'recordEnergyEval');
}

export async function runEnergyUpdate(state, target, options = {}) {
  return executeEnergyUpdate(null, state, target, options, 'runEnergyUpdate');
}

export async function recordEnergyUpdate(recorder, state, target, options = {}) {
  return executeEnergyUpdate(recorder, state, target, options, 'recordEnergyUpdate');
}

export async function runEnergyQuintelUpdate(state, options = {}) {
  return executeEnergyQuintelUpdate(null, state, options, 'runEnergyQuintelUpdate');
}

export async function runEnergyQuintelReduce(state, options = {}) {
  return executeEnergyQuintelReduce(null, state, options, 'runEnergyQuintelReduce');
}

export async function runEnergyQuintelGrad(state, options = {}) {
  return executeEnergyQuintelGrad(null, state, options, 'runEnergyQuintelGrad');
}

export async function recordEnergyQuintelUpdate(recorder, state, options = {}) {
  return executeEnergyQuintelUpdate(recorder, state, options, 'recordEnergyQuintelUpdate');
}

export async function recordEnergyQuintelGrad(recorder, state, options = {}) {
  return executeEnergyQuintelGrad(recorder, state, options, 'recordEnergyQuintelGrad');
}

export async function recordEnergyQuintelReduce(recorder, state, options = {}) {
  return executeEnergyQuintelReduce(recorder, state, options, 'recordEnergyQuintelReduce');
}
