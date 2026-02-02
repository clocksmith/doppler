
import { getDevice } from '../device.js';
import { acquireBuffer, getBufferRequestedSize } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { getPipelineFast, createUniformBufferWithView } from './utils.js';
import { trace } from '../../debug/index.js';
import { padToQ4KBlock } from '../../config/schema/index.js';
import { selectRuleValue } from './rule-registry.js';
import { selectRuleValue as selectLoaderRule } from '../../rules/rule-registry.js';


function canUseF16(input) {
  return input.dtype === 'f16';
}


function inferHiddenSize(input, hiddenSize) {
  if (hiddenSize != null) return hiddenSize;
  const shape = input?.shape;
  if (Array.isArray(shape) && shape.length > 0) {
    return shape[shape.length - 1];
  }
  return null;
}


function resolveParamDtype(options, hiddenSize) {
  if (options.normWeightDtype) {
    return selectLoaderRule('loader', 'weights', 'normWeightDtype', {
      normWeightDtype: options.normWeightDtype,
    });
  }
  const weight = options._weightBuffer;
  if (!weight || hiddenSize == null) {
    throw new Error('LayerNorm requires an explicit weight dtype or inferable weight buffer size.');
  }
  const byteSize = getBufferRequestedSize(weight);
  const f16Bytes = hiddenSize * 2;
  const f32Bytes = hiddenSize * 4;
  return selectLoaderRule('shared', 'dtype', 'dtypeFromSize', {
    bytesPerElement: byteSize >= f32Bytes ? 4 : byteSize >= f16Bytes ? 2 : 4,
  });
}


export function selectLayerNormKernel(options = {}, isF16 = false) {
  return selectRuleValue('layernorm', 'variant', { isF16 });
}


export async function runLayerNorm(
  input,
  weight,
  bias,
  eps,
  options = {}
) {
  const device = getDevice();
  if (eps == null) {
    throw new Error('LayerNorm requires an explicit eps value.');
  }

  const { batchSize = 1, hiddenSize = null, outputBuffer = null } = options;
  const isF16 = canUseF16(input);
  const variant = selectLayerNormKernel(options, isF16);
  trace.kernels(`LayerNorm: input.dtype=${input.dtype}, isF16=${isF16}, variant=${variant}`);

  const inferredHiddenSize = inferHiddenSize(input, hiddenSize);
  if (inferredHiddenSize == null) {
    throw new Error('LayerNorm requires hiddenSize or input shape to infer it.');
  }

  const paramsDtype = resolveParamDtype({ ...options, _weightBuffer: weight }, inferredHiddenSize);
  const constants = {
    PARAMS_IS_F16: paramsDtype === 'f16',
  };

  const pipeline = await getPipelineFast('layernorm', variant, null, constants);

  const bytesPerElement = isF16 ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(inferredHiddenSize);
  const outputSize = batchSize * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'layernorm_output');

  const uniformBuffer = createUniformBufferWithView(
    'layernorm_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredHiddenSize, true);
      view.setUint32(4, batchSize, true);
      view.setFloat32(8, eps, true);
      view.setUint32(12, 0, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'layernorm_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: weight } },
      { binding: 3, resource: { buffer: bias } },
      { binding: 4, resource: { buffer: outputBuf } },
    ],
  });

  dispatch(device, pipeline, bindGroup, batchSize, 'layernorm');

  uniformBuffer.destroy();

  return createTensor(outputBuf, input.dtype, [batchSize, inferredHiddenSize], 'layernorm_output');
}


export async function recordLayerNorm(
  recorder,
  input,
  weight,
  bias,
  eps,
  options = {}
) {
  const device = recorder.device;
  if (eps == null) {
    throw new Error('LayerNorm requires an explicit eps value.');
  }

  const { batchSize = 1, hiddenSize = null, outputBuffer = null } = options;
  const isF16 = canUseF16(input);
  const variant = selectLayerNormKernel(options, isF16);

  const inferredHiddenSize = inferHiddenSize(input, hiddenSize);
  if (inferredHiddenSize == null) {
    throw new Error('LayerNorm requires hiddenSize or input shape to infer it.');
  }

  const paramsDtype = resolveParamDtype({ ...options, _weightBuffer: weight }, inferredHiddenSize);
  const constants = {
    PARAMS_IS_F16: paramsDtype === 'f16',
  };

  const pipeline = await getPipelineFast('layernorm', variant, null, constants);

  const bytesPerElement = isF16 ? 2 : 4;
  const paddedHiddenSize = padToQ4KBlock(inferredHiddenSize);
  const outputSize = batchSize * paddedHiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'layernorm_output');

  const uniformBuffer = createUniformBufferWithView(
    'layernorm_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredHiddenSize, true);
      view.setUint32(4, batchSize, true);
      view.setFloat32(8, eps, true);
      view.setUint32(12, 0, true);
    },
    recorder
  );

  const bindGroup = device.createBindGroup({
    label: 'layernorm_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: weight } },
      { binding: 3, resource: { buffer: bias } },
      { binding: 4, resource: { buffer: outputBuf } },
    ],
  });

  recordDispatch(recorder, pipeline, bindGroup, batchSize, 'layernorm');

  return createTensor(outputBuf, input.dtype, [batchSize, inferredHiddenSize], 'layernorm_output');
}
