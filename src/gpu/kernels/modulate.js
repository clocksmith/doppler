import { getDevice } from '../device.js';
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView } from './utils.js';
import { selectRuleValue } from './rule-registry.js';

function selectModulateVariant(inputDtype, modDtype) {
  return selectRuleValue('modulate', 'variant', { inputDtype, modDtype });
}

export async function runModulate(input, mod, options = {}) {
  const device = getDevice();
  const {
    numTokens,
    hiddenSize,
    scaleOffset = 0,
    shiftOffset = 0,
    gateOffset = 0,
    hasGate = false,
    addOne = true,
    outputBuffer = null,
  } = options;

  if (!Number.isFinite(numTokens) || !Number.isFinite(hiddenSize)) {
    throw new Error('Modulate requires numTokens and hiddenSize.');
  }

  const variant = selectModulateVariant(input.dtype, mod.dtype);
  const pipeline = await createPipeline('modulate', variant);
  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = numTokens * hiddenSize * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'modulate_output');

  const uniformBuffer = createUniformBufferWithView(
    'modulate_uniforms',
    32,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, hiddenSize, true);
      view.setUint32(8, scaleOffset, true);
      view.setUint32(12, shiftOffset, true);
      view.setUint32(16, gateOffset, true);
      view.setUint32(20, hasGate ? 1 : 0, true);
      view.setUint32(24, addOne ? 1 : 0, true);
      view.setUint32(28, 0, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'modulate_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: mod.buffer } },
      { binding: 3, resource: { buffer: output } },
    ],
  });

  const workgroups = Math.ceil((numTokens * hiddenSize) / 256);
  dispatch(device, pipeline, bindGroup, workgroups, 'modulate');

  uniformBuffer.destroy();

  return createTensor(output, input.dtype, [numTokens, hiddenSize], 'modulate_output');
}

export async function recordModulate(recorder, input, mod, options = {}) {
  const device = recorder.device;
  const {
    numTokens,
    hiddenSize,
    scaleOffset = 0,
    shiftOffset = 0,
    gateOffset = 0,
    hasGate = false,
    addOne = true,
    outputBuffer = null,
  } = options;

  if (!Number.isFinite(numTokens) || !Number.isFinite(hiddenSize)) {
    throw new Error('Modulate requires numTokens and hiddenSize.');
  }

  const variant = selectModulateVariant(input.dtype, mod.dtype);
  const pipeline = await createPipeline('modulate', variant);
  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = numTokens * hiddenSize * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'modulate_output');

  const uniformBuffer = createUniformBufferWithView(
    'modulate_uniforms',
    32,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, hiddenSize, true);
      view.setUint32(8, scaleOffset, true);
      view.setUint32(12, shiftOffset, true);
      view.setUint32(16, gateOffset, true);
      view.setUint32(20, hasGate ? 1 : 0, true);
      view.setUint32(24, addOne ? 1 : 0, true);
      view.setUint32(28, 0, true);
    },
    recorder
  );

  const bindGroup = device.createBindGroup({
    label: 'modulate_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: mod.buffer } },
      { binding: 3, resource: { buffer: output } },
    ],
  });

  const workgroups = Math.ceil((numTokens * hiddenSize) / 256);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'modulate');

  return createTensor(output, input.dtype, [numTokens, hiddenSize], 'modulate_output');
}
