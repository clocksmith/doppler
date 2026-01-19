

import { getDevice, getKernelCapabilities } from '../device.js';
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { WORKGROUP_SIZES, VEC4_ELEMENTS_PER_WG } from './constants.js';
import { dispatch, dispatchIndirect, recordDispatch, recordDispatchIndirect } from './dispatch.js';
import { getPipelineFast, createUniformBufferWithView, getKernelConfig } from './utils.js';
import { trace } from '../../debug/index.js';
import { createTensor } from '../tensor.js';
import { DTYPE_SIZES } from '../../config/schema/index.js';
import { selectRuleValue as selectKernelRuleValue } from './rule-registry.js';
import { selectRuleValue as selectSharedRuleValue } from '../../rules/rule-registry.js';

function selectGatherVariant(useF16Input, useF16Output, useVec4) {
  return selectKernelRuleValue(
    'gather',
    'variant',
    { useF16Input, useF16Output, useVec4 }
  );
}


function getOutputBinding(variant, useF16Output) {
  if (!useF16Output) {
    return 3; // F32 output always uses binding 3
  }
  const config = getKernelConfig('gather', variant);
  const outputBinding = config.variantMetadata?.outputBinding;
  if (outputBinding == null) {
    throw new Error(`[Gather] Missing outputBinding for variant "${variant}" with f16 output.`);
  }
  return outputBinding;
}


export async function runGather(
  indices,
  embeddings,
  numTokens,
  hiddenSize,
  vocabSize,
  options = {}
) {
  const device = getDevice();
  const {
    useVec4 = true,
    outputBuffer = null,
    embeddingDtype,
    outputDtype,
    transpose = false,
    indexOffset = 0,
    indirectBuffer = null,
    indirectOffset = 0,
  } = options;

  // Detect embedding dtype (F16 embeddings enable optimized lm_head)
  const caps = getKernelCapabilities();
  if (embeddingDtype == null) {
    throw new Error('[Gather] embeddingDtype is required.');
  }
  if (outputDtype == null) {
    throw new Error('[Gather] outputDtype is required.');
  }
  const detectedDtype = embeddingDtype;
  const useF16Input = detectedDtype === 'f16' && caps.hasF16;
  const useF16Output = outputDtype === 'f16' && caps.hasF16;
  trace.embed(`Gather: numTokens=${numTokens}, hiddenSize=${hiddenSize}, vocabSize=${vocabSize}, transpose=${transpose}, indexOffset=${indexOffset}, detectedDtype=${detectedDtype}, useF16Input=${useF16Input}, useF16Output=${useF16Output}`);

  // Select kernel variant using lookup table
  const variant = selectGatherVariant(useF16Input, useF16Output, useVec4);
  trace.embed(`Gather variant: ${variant}`);
  const pipeline = await getPipelineFast('gather', variant);

  // Calculate output size using DTYPE_SIZES
  const outputDtypeKey = selectSharedRuleValue('shared', 'dtype', 'f16OrF32', { useF16: useF16Output });
  const bytesPerElement = DTYPE_SIZES[outputDtypeKey];
  const outputSize = numTokens * hiddenSize * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'gather_output');

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'gather_uniforms',
    32,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, hiddenSize, true);
      view.setUint32(8, vocabSize, true);
      view.setUint32(12, transpose ? 1 : 0, true);
      view.setUint32(16, indexOffset, true);
    },
    null,
    device
  );

  // Create bind group - output binding from kernel config
  const outputBinding = getOutputBinding(variant, useF16Output);
  
  const entries = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: indices } },
    { binding: 2, resource: { buffer: embeddings } },
    { binding: outputBinding, resource: { buffer: output } },
  ];
  const bindGroup = device.createBindGroup({
    label: 'gather_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  // gather.wgsl uses @workgroup_size(256); gather_vec4.wgsl uses @workgroup_size(64)
  // vec4 variant: 64 threads x 4 floats = 256 floats per workgroup
  const workgroups = useVec4
    ? Math.ceil((numTokens * hiddenSize) / VEC4_ELEMENTS_PER_WG)
    : Math.ceil((numTokens * hiddenSize) / WORKGROUP_SIZES.DEFAULT);
  if (indirectBuffer) {
    dispatchIndirect(device, pipeline, bindGroup, indirectBuffer, indirectOffset, 'gather');
  } else {
    dispatch(device, pipeline, bindGroup, workgroups, 'gather');
  }

  uniformBuffer.destroy();

  
  const actualDtype = selectSharedRuleValue('shared', 'dtype', 'f16OrF32', { useF16: useF16Output });
  return createTensor(output, actualDtype, [numTokens, hiddenSize], 'gather_output');
}


export async function recordGather(
  recorder,
  indices,
  embeddings,
  numTokens,
  hiddenSize,
  vocabSize,
  options = {}
) {
  const device = recorder.device;
  const {
    useVec4 = true,
    outputBuffer = null,
    embeddingDtype,
    outputDtype,
    transpose = false,
    indexOffset = 0,
    indirectBuffer = null,
    indirectOffset = 0,
  } = options;

  // Detect embedding dtype (same logic as runGather)
  const caps = getKernelCapabilities();
  if (embeddingDtype == null) {
    throw new Error('[Gather] embeddingDtype is required.');
  }
  if (outputDtype == null) {
    throw new Error('[Gather] outputDtype is required.');
  }
  const detectedDtype = embeddingDtype;
  const useF16Input = detectedDtype === 'f16' && caps.hasF16;
  const useF16Output = outputDtype === 'f16' && caps.hasF16;

  // Select kernel variant using lookup table
  const variant = selectGatherVariant(useF16Input, useF16Output, useVec4);
  trace.embed(`Gather variant: ${variant}`);
  const pipeline = await getPipelineFast('gather', variant);

  // Calculate output size using DTYPE_SIZES
  const outputDtypeKey = selectSharedRuleValue('shared', 'dtype', 'f16OrF32', { useF16: useF16Output });
  const bytesPerElement = DTYPE_SIZES[outputDtypeKey];
  const outputSize = numTokens * hiddenSize * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'gather_output');

  // Uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'gather_uniforms',
    32,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, hiddenSize, true);
      view.setUint32(8, vocabSize, true);
      view.setUint32(12, transpose ? 1 : 0, true);
      view.setUint32(16, indexOffset, true);
    },
    recorder
  );

  // Create bind group - output binding from kernel config
  const outputBinding = getOutputBinding(variant, useF16Output);
  
  const entries = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: indices } },
    { binding: 2, resource: { buffer: embeddings } },
    { binding: outputBinding, resource: { buffer: output } },
  ];
  const bindGroup = device.createBindGroup({
    label: 'gather_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  // gather.wgsl uses @workgroup_size(256); gather_vec4.wgsl uses @workgroup_size(64)
  // vec4 variant: 64 threads x 4 floats = 256 floats per workgroup
  const workgroups = useVec4
    ? Math.ceil((numTokens * hiddenSize) / VEC4_ELEMENTS_PER_WG)
    : Math.ceil((numTokens * hiddenSize) / WORKGROUP_SIZES.DEFAULT);
  if (indirectBuffer) {
    recordDispatchIndirect(recorder, pipeline, bindGroup, indirectBuffer, indirectOffset, 'gather');
  } else {
    recordDispatch(recorder, pipeline, bindGroup, workgroups, 'gather');
  }

  
  const actualDtype = selectSharedRuleValue('shared', 'dtype', 'f16OrF32', { useF16: useF16Output });
  return createTensor(output, actualDtype, [numTokens, hiddenSize], 'gather_output');
}
