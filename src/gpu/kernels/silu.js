import { executePairedSiLU } from './paired-activation.js';


import { getDevice } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { getPipelineFast, getPipelineBindGroupLayout } from './pipeline-cache.js';
import { createUniformBufferWithView } from './uniform-utils.js';
import { selectRuleValue } from './rule-registry.js';

function destroyAfterSubmit(device, buffer) {
  if (!buffer) {
    return;
  }
  device.queue.onSubmittedWorkDone()
    .then(() => {
      buffer.destroy();
    })
    .catch(() => {
      buffer.destroy();
    });
}

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

function resolveInputActivation(value, hasGate) {
  if (value === undefined) {
    if (hasGate) {
      throw new Error('Gated SiLU requires an explicit inputActivation (identity or silu).');
    }
    return 'silu';
  }
  if (value !== 'identity' && value !== 'silu') {
    throw new Error(`Unsupported SiLU inputActivation: ${value}`);
  }
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

function cleanupRunResources(uniformBuffer, ownedOutput) {
  if (ownedOutput) {
    releaseBuffer(ownedOutput);
  }
}

function planSiLUDispatch(device, size, useVec4) {
  const maxPerDim = Number.isFinite(device?.limits?.maxComputeWorkgroupsPerDimension)
    ? device.limits.maxComputeWorkgroupsPerDimension
    : 65535;
  const laneWidth = useVec4 ? 4 : 1;
  const chunkSize = maxPerDim * WORKGROUP_SIZES.DEFAULT * laneWidth;
  const dispatchStride = Math.min(size, chunkSize);
  const x = Math.min(maxPerDim, Math.ceil(dispatchStride / (WORKGROUP_SIZES.DEFAULT * laneWidth)));
  const y = Math.max(1, Math.ceil(size / chunkSize));
  return { dispatchStride, workgroups: [x, y, 1] };
}


async function executeSiLU(
  recorder,
  input,
  options = {}
) {
  if (input.precision !== undefined || input.components !== undefined || input.elementStrideBytes !== undefined
    || options.gate?.precision !== undefined || options.gate?.components !== undefined || options.gate?.elementStrideBytes !== undefined) {
    return executePairedSiLU(recorder, input, options.size ?? input.shape.reduce((total, size) => total * size, 1), {
      ...options, gateActivation: options.gateActivation ?? 'silu',
      inputActivation: resolveInputActivation(options.inputActivation, Boolean(options.gate)),
    });
  }

  const device = recorder ? recorder.device : getDevice();
  const {
    size,
    gate = null,
    outputBuffer = null,
    useVec4: requestedVec4 = false,
    swigluLimit,
    gateActivation = 'silu',
    inputActivation,
  } = options;
  const resolvedSwigluLimit = resolveSwigluLimit(swigluLimit, 'SiLU');
  const resolvedInputActivation = resolveInputActivation(inputActivation, Boolean(gate));

  const isF16 = canUseF16(input);
  const bytesPerElement = dtypeBytes(input.dtype);

  // Select variant using lookup table
  const variant = selectSiLUVariant(isF16);
  const useVec4 = recorder ? false : requestedVec4;
  const overrides = resolveOverrides('silu', {
    hasGate: Boolean(gate),
    useVec4,
    useSplit: false,
    useRowsplit: false,
  });
  const constants = {
    ...(overrides || {}),
    ...(gate && gateActivation === 'sigmoid' ? { GATE_USE_SIGMOID: true } : {}),
    ...(resolvedInputActivation === 'identity' ? { INPUT_USE_IDENTITY: true } : {}),
  };
  const pipeline = await getPipelineFast('silu', variant, null, constants);

  const inferredSize = size || (input.buffer.size / bytesPerElement);
  const outputSize = inferredSize * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'silu_output');
  const ownedOutput = outputBuffer ? null : output;
  const dispatchPlan = planSiLUDispatch(device, inferredSize, useVec4);

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'silu_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredSize, true);
      view.setUint32(4, dispatchPlan.dispatchStride, true);
      view.setFloat32(8, gate ? resolvedSwigluLimit : 0, true);
      view.setFloat32(12, 0, true);
    },
    recorder,
    device
  );

  // Create bind group using helper
  const entries = createSiLUBindGroupEntries(uniformBuffer, input, output, gate);

  try {
    const descriptor = {
      label: 'silu_bind_group',
      layout: getPipelineBindGroupLayout(pipeline, 0),
      entries,
    };
    const bindGroup = recorder?.decodeWorkspace
      ? recorder.decodeWorkspace.getBindGroupForOutputs(`silu:${variant}`, [output], descriptor)
      : device.createBindGroup(descriptor);

    if (recorder) recordDispatch(recorder, pipeline, bindGroup, dispatchPlan.workgroups, 'silu');
    else dispatch(device, pipeline, bindGroup, dispatchPlan.workgroups, 'silu');
    return createTensor(output, input.dtype, [inferredSize], 'silu_output');
  } catch (error) {
    cleanupRunResources(null, ownedOutput);
    throw error;
  } finally {
    if (!recorder) destroyAfterSubmit(device, uniformBuffer);
  }
}

export async function runSiLU(
  input,
  options = {}
) {
  return executeSiLU(null, input, options);
}


export async function runSwiGLURowsplitBias(
  input,
  bias,
  numTokens,
  dim,
  options = {}
) {
  if (input.precision !== undefined) throw new Error('[SiLU] Paired row-split execution requires a declared paired kernel.');
  const device = getDevice();
  const { outputBuffer = null, biasOffset = 0, swigluLimit } = options;
  const resolvedSwigluLimit = resolveSwigluLimit(swigluLimit, 'SwiGLU row-split');

  const useF16 = input.dtype === 'f16' && bias.dtype === 'f16';
  const variant = selectSwiGLURowsplitBiasVariant(useF16);
  const pipeline = await getPipelineFast('swiglu', variant);

  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = numTokens * dim * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'swiglu_output');
  const ownedOutput = outputBuffer ? null : output;

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
  try {
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
    return createTensor(output, input.dtype, [numTokens, dim], 'swiglu_output');
  } catch (error) {
    cleanupRunResources(null, ownedOutput);
    throw error;
  } finally {
    destroyAfterSubmit(device, uniformBuffer);
  }
}


async function executeSiLURowSplit(
  recorder,
  input,
  options
) {
  if (input.precision !== undefined) throw new Error('[SiLU] Paired row-split execution requires a declared paired kernel.');
  const device = recorder ? recorder.device : getDevice();
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
  const ownedOutput = outputBuffer ? null : output;

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
    recorder,
    device
  );

  // Bind group: provide a dummy gate buffer to satisfy the fixed layout
  try {
    const gateBuffer = input.buffer;
    const descriptor = {
      label: 'silu_rowsplit_bind_group',
      layout: getPipelineBindGroupLayout(pipeline, 0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: input.buffer } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: gateBuffer } },
      ],
    };
    const bindGroup = recorder?.decodeWorkspace
      ? recorder.decodeWorkspace.getBindGroupForOutputs(`silu-rowsplit:${variant}`, [output], descriptor)
      : device.createBindGroup(descriptor);

    const workgroups = [Math.ceil(dim / WORKGROUP_SIZES.DEFAULT), numTokens, 1];
    if (recorder) recordDispatch(recorder, pipeline, bindGroup, workgroups, 'silu_rowsplit');
    else dispatch(device, pipeline, bindGroup, workgroups, 'silu_rowsplit');
    return createTensor(output, input.dtype, [numTokens, dim], 'silu_rowsplit_output');
  } catch (error) {
    cleanupRunResources(null, ownedOutput);
    throw error;
  } finally {
    if (!recorder) uniformBuffer.destroy();
  }
}

export async function runSiLURowSplit(
  input,
  options
) {
  return executeSiLURowSplit(null, input, options);
}


export async function recordSiLURowSplit(
  recorder,
  input,
  options
) {
  return executeSiLURowSplit(recorder, input, options);
}


export async function recordSiLU(
  recorder,
  input,
  options = {}
) {
  return executeSiLU(recorder, input, options);
}
