
import { getDevice, getKernelCapabilities } from '../device.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView, getKernelConfig, hasRequiredFeatures } from './utils.js';
import { selectRuleValue as selectKernelRuleValue } from './rule-registry.js';


function resolveQuantizeVariant(mode) {
  return selectKernelRuleValue('kv_quantize', 'variant', { mode });
}


function createQuantizeUniformBuffer(device, recorder, params) {
  return createUniformBufferWithView(
    'kv_quantize_uniforms',
    32,
    (view) => {
      view.setUint32(0, params.numKVHeads, true);
      view.setUint32(4, params.headDim, true);
      view.setUint32(8, params.startPos, true);
      view.setUint32(12, params.numTokens, true);
      view.setUint32(16, params.packedStride, true);
      view.setUint32(20, 0, true);
    },
    recorder,
    device
  );
}


export async function runKVQuantize(
  keys,
  values,
  outputKeys,
  outputValues,
  scalesK,
  scalesV,
  options = {}
) {
  const device = getDevice();
  const {
    numKVHeads,
    headDim,
    startPos,
    numTokens,
    packedStride,
    mode = 'int8',
  } = options;

  const variant = resolveQuantizeVariant(mode);
  const config = getKernelConfig('kv_quantize', variant);
  const caps = getKernelCapabilities();
  if (!hasRequiredFeatures(config.requires, caps)) {
    throw new Error(`KV quantize kernel "${variant}" requires unsupported GPU features.`);
  }

  const pipeline = await createPipeline('kv_quantize', variant);
  const uniformBuffer = createQuantizeUniformBuffer(device, null, {
    numKVHeads,
    headDim,
    startPos,
    numTokens,
    packedStride,
  });

  const bindGroup = device.createBindGroup({
    label: 'kv_quantize_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: keys } },
      { binding: 2, resource: { buffer: values } },
      { binding: 3, resource: { buffer: outputKeys } },
      { binding: 4, resource: { buffer: outputValues } },
      { binding: 5, resource: { buffer: scalesK } },
      { binding: 6, resource: { buffer: scalesV } },
    ],
  });

  const workgroups = [numKVHeads, numTokens, 1];
  dispatch(device, pipeline, bindGroup, workgroups, 'kv_quantize');
  uniformBuffer.destroy();
}


export async function recordKVQuantize(
  recorder,
  keys,
  values,
  outputKeys,
  outputValues,
  scalesK,
  scalesV,
  options = {}
) {
  const device = recorder.device;
  const {
    numKVHeads,
    headDim,
    startPos,
    numTokens,
    packedStride,
    mode = 'int8',
  } = options;

  const variant = resolveQuantizeVariant(mode);
  const config = getKernelConfig('kv_quantize', variant);
  const caps = getKernelCapabilities();
  if (!hasRequiredFeatures(config.requires, caps)) {
    throw new Error(`KV quantize kernel "${variant}" requires unsupported GPU features.`);
  }

  const pipeline = await createPipeline('kv_quantize', variant);
  const uniformBuffer = createQuantizeUniformBuffer(device, recorder, {
    numKVHeads,
    headDim,
    startPos,
    numTokens,
    packedStride,
  });

  const bindGroup = device.createBindGroup({
    label: 'kv_quantize_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: keys } },
      { binding: 2, resource: { buffer: values } },
      { binding: 3, resource: { buffer: outputKeys } },
      { binding: 4, resource: { buffer: outputValues } },
      { binding: 5, resource: { buffer: scalesK } },
      { binding: 6, resource: { buffer: scalesV } },
    ],
  });

  const workgroups = [numKVHeads, numTokens, 1];
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'kv_quantize');
}
