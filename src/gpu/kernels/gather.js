

import { getDevice, getKernelCapabilities } from '../device.js';
import { acquireBuffer } from '../buffer-pool.js';
import { WORKGROUP_SIZES, VEC4_ELEMENTS_PER_WG } from './constants.js';
import { dispatch, dispatchIndirect, recordDispatch, recordDispatchIndirect } from './dispatch.js';
import { getPipelineFast, createUniformBufferWithView, getKernelConfig } from './utils.js';
import { trace } from '../../debug/index.js';
import { createTensor } from '../tensor.js';
import { DTYPE_SIZES } from '../../config/schema/index.js';
import { selectByRules } from './rule-matcher.js';

function selectGatherVariant(useF16Input, useF16Output, useVec4) {
  const rules = [
    { match: { useF16Input: true, useF16Output: true, useVec4: true }, value: 'f16_vec4_f16_out' },
    { match: { useF16Input: true, useF16Output: true }, value: 'f16_f16_out' },
    { match: { useF16Input: true, useVec4: true }, value: 'f16_vec4' },
    { match: { useF16Input: true }, value: 'f16' },
    { match: { useF16Output: true, useVec4: true }, value: 'vec4_f16_out' },
    { match: { useF16Output: true }, value: 'f16_out' },
    { match: { useVec4: true }, value: 'vec4' },
    { match: {}, value: 'default' },
  ];

  return selectByRules(
    rules,
    { useF16Input, useF16Output, useVec4 },
    'default'
  );
}


function getOutputBinding(variant, useF16Output) {
  if (!useF16Output) {
    return 3; // F32 output always uses binding 3
  }
  const config = getKernelConfig('gather', variant);
  return config.variantMetadata?.outputBinding ?? 4;
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
    outputDtype = 'f32',
    transpose = false,
    indexOffset = 0,
    indirectBuffer = null,
    indirectOffset = 0,
  } = options;

  // Detect embedding dtype (F16 embeddings enable optimized lm_head)
  const caps = getKernelCapabilities();
  const detectedDtype = embeddingDtype || 'f32';
  const useF16Input = detectedDtype === 'f16' && caps.hasF16;
  const useF16Output = outputDtype === 'f16' && caps.hasF16;
  trace.embed(`Gather: numTokens=${numTokens}, hiddenSize=${hiddenSize}, vocabSize=${vocabSize}, transpose=${transpose}, indexOffset=${indexOffset}, detectedDtype=${detectedDtype}, useF16Input=${useF16Input}, useF16Output=${useF16Output}`);

  // Select kernel variant using lookup table
  const variant = selectGatherVariant(useF16Input, useF16Output, useVec4);
  trace.embed(`Gather variant: ${variant}`);
  const pipeline = await getPipelineFast('gather', variant);

  // Calculate output size using DTYPE_SIZES
  const outputDtypeKey = useF16Output ? 'f16' : 'f32';
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

  
  const actualDtype = useF16Output ? 'f16' : 'f32';
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
    outputDtype = 'f32',
    transpose = false,
    indexOffset = 0,
    indirectBuffer = null,
    indirectOffset = 0,
  } = options;

  // Detect embedding dtype (same logic as runGather)
  const caps = getKernelCapabilities();
  const detectedDtype = embeddingDtype || 'f32';
  const useF16Input = detectedDtype === 'f16' && caps.hasF16;
  const useF16Output = outputDtype === 'f16' && caps.hasF16;

  // Select kernel variant using lookup table
  const variant = selectGatherVariant(useF16Input, useF16Output, useVec4);
  trace.embed(`Gather variant: ${variant}`);
  const pipeline = await getPipelineFast('gather', variant);

  // Calculate output size using DTYPE_SIZES
  const outputDtypeKey = useF16Output ? 'f16' : 'f32';
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

  
  const actualDtype = useF16Output ? 'f16' : 'f32';
  return createTensor(output, actualDtype, [numTokens, hiddenSize], 'gather_output');
}
