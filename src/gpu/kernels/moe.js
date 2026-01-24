

import { getDevice } from '../device.js';
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { WORKGROUP_SIZES, GPU_LIMITS } from './constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView, createBindGroupWithValidation } from './utils.js';
import { trace } from '../../debug/index.js';
import { selectRuleValue as selectKernelRuleValue } from './rule-registry.js';
import { selectRuleValue as selectSharedRuleValue } from '../../rules/rule-registry.js';


function calculate2DDispatch(totalWorkgroups) {
  const maxWG = GPU_LIMITS.MAX_WORKGROUPS;
  if (totalWorkgroups <= maxWG) {
    return {
      x: totalWorkgroups,
      y: 1,
      threadsPerRow: totalWorkgroups * WORKGROUP_SIZES.DEFAULT,
    };
  }
  // Split across X and Y dimensions
  const x = maxWG;
  const y = Math.ceil(totalWorkgroups / maxWG);
  return {
    x,
    y,
    threadsPerRow: x * WORKGROUP_SIZES.DEFAULT,
  };
}


export async function runTopK(probs, numTokens, numExperts, topK, options = {}) {
  const device = getDevice();
  const { normalize = true } = options;

  const pipeline = await createPipeline('topk', 'default');

  // Output buffers
  const indicesSize = numTokens * topK * 4; // u32
  const weightsSize = numTokens * topK * 4; // f32
  const indices = acquireBuffer(indicesSize, undefined, 'topk_indices');
  const weights = acquireBuffer(weightsSize, undefined, 'topk_weights');

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'topk_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, numExperts, true);
      view.setUint32(8, topK, true);
      view.setUint32(12, normalize ? 1 : 0, true);
    },
    null,
    device
  );

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'topk_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: probs } },
      { binding: 2, resource: { buffer: indices } },
      { binding: 3, resource: { buffer: weights } },
    ],
  });

  dispatch(device, pipeline, bindGroup, numTokens, 'topk');

  uniformBuffer.destroy();

  return { indices, weights };
}


export async function recordTopK(recorder, probs, numTokens, numExperts, topK, options = {}) {
  const device = recorder.device;
  const { normalize = true } = options;

  const pipeline = await createPipeline('topk', 'default');

  const indicesSize = numTokens * topK * 4; // u32
  const weightsSize = numTokens * topK * 4; // f32
  const indices = acquireBuffer(indicesSize, undefined, 'topk_indices');
  const weights = acquireBuffer(weightsSize, undefined, 'topk_weights');

  const uniformBuffer = createUniformBufferWithView(
    'topk_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, numExperts, true);
      view.setUint32(8, topK, true);
      view.setUint32(12, normalize ? 1 : 0, true);
    },
    recorder
  );

  const bindGroup = device.createBindGroup({
    label: 'topk_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: probs } },
      { binding: 2, resource: { buffer: indices } },
      { binding: 3, resource: { buffer: weights } },
    ],
  });

  recordDispatch(recorder, pipeline, bindGroup, numTokens, 'topk');

  return { indices, weights };
}

// Cached explicit bind group layout for MoE gather (all 6 bindings).
// Internal postmortems cover why this explicit layout is required.

let moeGatherBindGroupLayout = null;


function getMoEGatherBindGroupLayout(device) {
  if (moeGatherBindGroupLayout) return moeGatherBindGroupLayout;

  moeGatherBindGroupLayout = device.createBindGroupLayout({
    label: 'moe_gather_explicit_layout',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  return moeGatherBindGroupLayout;
}

// Cached explicit bind group layout for scatter-add dynamic (all 6 bindings)
// Required because auto layout can omit bindings in some driver/compiler paths.
let scatterAddDynamicBindGroupLayout = null;

function getScatterAddDynamicBindGroupLayout(device) {
  if (scatterAddDynamicBindGroupLayout) return scatterAddDynamicBindGroupLayout;

  scatterAddDynamicBindGroupLayout = device.createBindGroupLayout({
    label: 'scatter_add_dynamic_explicit_layout',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  return scatterAddDynamicBindGroupLayout;
}


export async function runMoEGather(hiddenStates, expertIndices, numTokens, hiddenSize, numExperts, topK, options = {}) {
  const device = getDevice();
  const { maxTokensPerExpert = numTokens } = options;
  const useF16 = hiddenStates.dtype === 'f16';
  const suffix = selectKernelRuleValue('moe', 'variantSuffix', { useF16 });
  const dtypeLabel = selectSharedRuleValue('shared', 'dtype', 'f16OrF32', { useF16 });

  // Use explicit bind group layout (required because count_and_map doesn't use all bindings)
  const explicitLayout = getMoEGatherBindGroupLayout(device);

  // Two-phase approach: count_and_map builds token assignments, gather copies hidden states
  const countPipeline = await createPipeline('moe_gather', `count${suffix}`, explicitLayout);
  const gatherPipeline = await createPipeline('moe_gather', `gather${suffix}`, explicitLayout);

  // Output buffers per WGSL shader:
  // - gathered: [numExperts, maxTokensPerExpert, hiddenSize]
  // - tokenCounts: [numExperts]
  // - tokenMap: [numExperts, maxTokensPerExpert, 2] (tokenIdx, kIdx)
  const bytesPerElement = hiddenStates.dtype === 'f16' ? 2 : 4;
  const gatheredSize = numExperts * maxTokensPerExpert * hiddenSize * bytesPerElement;

  // Calculate 2D dispatch for gather phase (may exceed 65535 workgroups)
  const gatherWorkgroupsTotal = Math.ceil((numExperts * maxTokensPerExpert * hiddenSize) / WORKGROUP_SIZES.DEFAULT);
  const gatherDispatch = calculate2DDispatch(gatherWorkgroupsTotal);

  trace.kernels('moe_gather params', { numTokens, hiddenSize, numExperts, topK, maxTokensPerExpert, gatheredSize, gatherWorkgroups: gatherWorkgroupsTotal, gatherDispatch });
  const tokenCountsSize = numExperts * 4;
  const tokenMapSize = numExperts * maxTokensPerExpert * 2 * 4;

  const gatheredBuffer = acquireBuffer(gatheredSize, undefined, 'moe_gathered');
  const tokenCounts = acquireBuffer(tokenCountsSize, undefined, 'moe_token_counts');
  const tokenMap = acquireBuffer(tokenMapSize, undefined, 'moe_token_map');

  // Create uniform buffer (32 bytes to match WGSL struct with padding)
  const uniformBuffer = createUniformBufferWithView(
    'moe_gather_uniforms',
    32,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, hiddenSize, true);
      view.setUint32(8, numExperts, true);
      view.setUint32(12, topK, true);
      view.setUint32(16, maxTokensPerExpert, true);
      view.setUint32(20, gatherDispatch.threadsPerRow, true); // threads_per_row for 2D dispatch
      view.setUint32(24, 0, true); // _pad2
      view.setUint32(28, 0, true); // _pad3
    },
    null,
    device
  );

  // Create bind group with explicit layout (all 6 bindings)
  const bindGroup = await createBindGroupWithValidation(device, {
    label: 'moe_gather_bind_group',
    layout: explicitLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: hiddenStates.buffer } },
      { binding: 2, resource: { buffer: expertIndices } },
      { binding: 3, resource: { buffer: gatheredBuffer } },
      { binding: 4, resource: { buffer: tokenCounts } },
      { binding: 5, resource: { buffer: tokenMap } },
    ],
  }, `moe_gather:${dtypeLabel}`);

  // Phase 1: Count tokens per expert and build token map
  const encoder = device.createCommandEncoder({ label: 'moe_gather_encoder' });
  encoder.clearBuffer(tokenCounts); // Zero-initialize tokenCounts (atomics start at 0)

  const countPass = encoder.beginComputePass({ label: 'moe_gather_count_pass' });
  countPass.setPipeline(countPipeline);
  countPass.setBindGroup(0, bindGroup);
  const countWorkgroups = Math.ceil((numTokens * topK) / WORKGROUP_SIZES.DEFAULT);
  countPass.dispatchWorkgroups(countWorkgroups);
  countPass.end();

  // Phase 2: Gather hidden states based on token map (2D dispatch for large workgroup counts)
  const gatherPass = encoder.beginComputePass({ label: 'moe_gather_gather_pass' });
  gatherPass.setPipeline(gatherPipeline);
  gatherPass.setBindGroup(0, bindGroup);
  gatherPass.dispatchWorkgroups(gatherDispatch.x, gatherDispatch.y, 1);
  gatherPass.end();

  device.queue.submit([encoder.finish()]);

  uniformBuffer.destroy();

  const gathered = createTensor(
    gatheredBuffer,
    hiddenStates.dtype,
    [numExperts, maxTokensPerExpert, hiddenSize],
    'moe_gathered'
  );

  return { gathered, tokenCounts, tokenMap, maxTokensPerExpert };
}


export async function recordMoEGather(recorder, hiddenStates, expertIndices, numTokens, hiddenSize, numExperts, topK, options = {}) {
  const device = recorder.device;
  const { maxTokensPerExpert = numTokens } = options;
  const useF16 = hiddenStates.dtype === 'f16';
  const suffix = selectKernelRuleValue('moe', 'variantSuffix', { useF16 });
  const dtypeLabel = selectSharedRuleValue('shared', 'dtype', 'f16OrF32', { useF16 });

  // Use explicit bind group layout (required because count_and_map doesn't use all bindings)
  const explicitLayout = getMoEGatherBindGroupLayout(device);

  // Two-phase approach: count_and_map builds token assignments, gather copies hidden states
  const countPipeline = await createPipeline('moe_gather', `count${suffix}`, explicitLayout);
  const gatherPipeline = await createPipeline('moe_gather', `gather${suffix}`, explicitLayout);

  const bytesPerElement = hiddenStates.dtype === 'f16' ? 2 : 4;
  const gatheredSize = numExperts * maxTokensPerExpert * hiddenSize * bytesPerElement;

  // Calculate 2D dispatch for gather phase (may exceed 65535 workgroups)
  const gatherWorkgroupsTotal = Math.ceil((numExperts * maxTokensPerExpert * hiddenSize) / WORKGROUP_SIZES.DEFAULT);
  const gatherDispatch = calculate2DDispatch(gatherWorkgroupsTotal);

  const tokenCountsSize = numExperts * 4;
  const tokenMapSize = numExperts * maxTokensPerExpert * 2 * 4;

  const gatheredBuffer = acquireBuffer(gatheredSize, undefined, 'moe_gathered');
  const tokenCounts = acquireBuffer(tokenCountsSize, undefined, 'moe_token_counts');
  const tokenMap = acquireBuffer(tokenMapSize, undefined, 'moe_token_map');

  // Create uniform buffer (32 bytes to match WGSL struct with padding)
  const uniformBuffer = createUniformBufferWithView(
    'moe_gather_uniforms',
    32,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, hiddenSize, true);
      view.setUint32(8, numExperts, true);
      view.setUint32(12, topK, true);
      view.setUint32(16, maxTokensPerExpert, true);
      view.setUint32(20, gatherDispatch.threadsPerRow, true); // threads_per_row for 2D dispatch
      view.setUint32(24, 0, true); // _pad2
      view.setUint32(28, 0, true); // _pad3
    },
    recorder
  );

  // Create bind group with explicit layout (all 6 bindings)
  const bindGroup = await createBindGroupWithValidation(device, {
    label: 'moe_gather_bind_group',
    layout: explicitLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: hiddenStates.buffer } },
      { binding: 2, resource: { buffer: expertIndices } },
      { binding: 3, resource: { buffer: gatheredBuffer } },
      { binding: 4, resource: { buffer: tokenCounts } },
      { binding: 5, resource: { buffer: tokenMap } },
    ],
  }, `moe_gather:${dtypeLabel}`);

  const encoder = recorder.getEncoder();
  encoder.clearBuffer(tokenCounts);

  // Phase 1: Count tokens per expert and build token map
  const countPass = recorder.beginComputePass('moe_gather_count');
  countPass.setPipeline(countPipeline);
  countPass.setBindGroup(0, bindGroup);
  countPass.dispatchWorkgroups(Math.ceil((numTokens * topK) / WORKGROUP_SIZES.DEFAULT));
  countPass.end();

  // Phase 2: Gather hidden states based on token map (2D dispatch for large workgroup counts)
  const gatherPass = recorder.beginComputePass('moe_gather_gather');
  gatherPass.setPipeline(gatherPipeline);
  gatherPass.setBindGroup(0, bindGroup);
  gatherPass.dispatchWorkgroups(gatherDispatch.x, gatherDispatch.y, 1);
  gatherPass.end();

  const gathered = createTensor(
    gatheredBuffer,
    hiddenStates.dtype,
    [numExperts, maxTokensPerExpert, hiddenSize],
    'moe_gathered'
  );

  return { gathered, tokenCounts, tokenMap, maxTokensPerExpert };
}


export async function runScatterAdd(expertOutputs, indices, weights, numTokens, hiddenSize, numExperts, topK, options = {}) {
  const device = getDevice();
  const { outputBuffer = null } = options;

  const pipeline = await createPipeline('scatter_add', 'default');

  // Output: [numTokens, hiddenSize]
  const bytesPerElement = expertOutputs.dtype === 'f16' ? 2 : 4;
  const outputSize = numTokens * hiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'scatter_add_output');

  // Create uniform buffer
  // WGSL struct order: numTokens, hiddenSize, topK, numExperts
  const uniformBuffer = createUniformBufferWithView(
    'scatter_add_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, hiddenSize, true);
      view.setUint32(8, topK, true);      // offset 8 = topK (per WGSL struct)
      view.setUint32(12, numExperts, true); // offset 12 = numExperts
    },
    null,
    device
  );

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'scatter_add_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: expertOutputs.buffer } },
      { binding: 2, resource: { buffer: indices } },
      { binding: 3, resource: { buffer: weights } },
      { binding: 4, resource: { buffer: outputBuf } },
    ],
  });

  // Dispatch
  const encoder = device.createCommandEncoder({ label: 'scatter_add_encoder' });
  encoder.clearBuffer(outputBuf);
  const pass = encoder.beginComputePass({ label: 'scatter_add_pass' });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);

  // WGSL main kernel: each thread handles one output element (numTokens * hiddenSize total)
  const workgroups = Math.ceil((numTokens * hiddenSize) / WORKGROUP_SIZES.DEFAULT);
  pass.dispatchWorkgroups(workgroups);
  pass.end();

  device.queue.submit([encoder.finish()]);

  uniformBuffer.destroy();

  return createTensor(outputBuf, expertOutputs.dtype, [numTokens, hiddenSize], 'scatter_add_output');
}


export async function recordScatterAdd(recorder, expertOutputs, indices, weights, numTokens, hiddenSize, numExperts, topK, options = {}) {
  const device = recorder.device;
  const { outputBuffer = null } = options;

  const pipeline = await createPipeline('scatter_add', 'default');
  const bytesPerElement = expertOutputs.dtype === 'f16' ? 2 : 4;
  const outputSize = numTokens * hiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'scatter_add_output');

  // WGSL struct order: numTokens, hiddenSize, topK, numExperts
  const uniformBuffer = createUniformBufferWithView(
    'scatter_add_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, hiddenSize, true);
      view.setUint32(8, topK, true);      // offset 8 = topK (per WGSL struct)
      view.setUint32(12, numExperts, true); // offset 12 = numExperts
    },
    recorder
  );

  const bindGroup = device.createBindGroup({
    label: 'scatter_add_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: expertOutputs.buffer } },
      { binding: 2, resource: { buffer: indices } },
      { binding: 3, resource: { buffer: weights } },
      { binding: 4, resource: { buffer: outputBuf } },
    ],
  });

  recorder.getEncoder().clearBuffer(outputBuf);

  const pass = recorder.beginComputePass('scatter_add');
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  // WGSL main kernel: each thread handles one output element (numTokens * hiddenSize total)
  pass.dispatchWorkgroups(Math.ceil((numTokens * hiddenSize) / WORKGROUP_SIZES.DEFAULT));
  pass.end();

  return createTensor(outputBuf, expertOutputs.dtype, [numTokens, hiddenSize], 'scatter_add_output');
}


export async function runScatterAddDynamic(expertOutputs, indices, weights, tokenOffsets, numTokens, hiddenSize, topK, options = {}) {
  const device = getDevice();
  const { outputBuffer = null, weightsDtype = 'f32' } = options;

  if (weightsDtype === 'f16' && expertOutputs.dtype !== 'f16') {
    throw new Error('ScatterAddDynamic f16 weights require f16 expert outputs');
  }

  const variant = selectKernelRuleValue('moe', 'scatterAddVariant', {
    outputDtype: expertOutputs.dtype,
    weightsDtype,
  });
  const explicitLayout = getScatterAddDynamicBindGroupLayout(device);
  const pipeline = await createPipeline('scatter_add', variant, explicitLayout);

  // Output: [numTokens, hiddenSize]
  const bytesPerElement = expertOutputs.dtype === 'f16' ? 2 : 4;
  const outputSize = numTokens * hiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'scatter_add_dynamic_output');

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'scatter_add_dynamic_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, hiddenSize, true);
      view.setUint32(8, topK, true);
    },
    null,
    device
  );

  // Create bind group
  const bindGroup = await createBindGroupWithValidation(device, {
    label: 'scatter_add_dynamic_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: expertOutputs.buffer } },
      { binding: 2, resource: { buffer: indices } },
      { binding: 3, resource: { buffer: weights } },
      { binding: 4, resource: { buffer: tokenOffsets } },
      { binding: 5, resource: { buffer: outputBuf } },
    ],
  }, `scatter_add_dynamic:${variant}`);

  // Dispatch
  const encoder = device.createCommandEncoder({ label: 'scatter_add_dynamic_encoder' });
  encoder.clearBuffer(outputBuf);
  const pass = encoder.beginComputePass({ label: 'scatter_add_dynamic_pass' });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);

  const workgroups = Math.ceil((numTokens * topK * hiddenSize) / WORKGROUP_SIZES.DEFAULT);
  pass.dispatchWorkgroups(workgroups);
  pass.end();

  device.queue.submit([encoder.finish()]);

  uniformBuffer.destroy();

  return createTensor(outputBuf, expertOutputs.dtype, [numTokens, hiddenSize], 'scatter_add_dynamic_output');
}


export async function recordScatterAddDynamic(recorder, expertOutputs, indices, weights, tokenOffsets, numTokens, hiddenSize, topK, options = {}) {
  const device = recorder.device;
  const { outputBuffer = null, weightsDtype = 'f32' } = options;

  if (weightsDtype === 'f16' && expertOutputs.dtype !== 'f16') {
    throw new Error('ScatterAddDynamic f16 weights require f16 expert outputs');
  }

  const variant = selectKernelRuleValue('moe', 'scatterAddVariant', {
    outputDtype: expertOutputs.dtype,
    weightsDtype,
  });
  const explicitLayout = getScatterAddDynamicBindGroupLayout(device);
  const pipeline = await createPipeline('scatter_add', variant, explicitLayout);
  const bytesPerElement = expertOutputs.dtype === 'f16' ? 2 : 4;
  const outputSize = numTokens * hiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'scatter_add_dynamic_output');

  const uniformBuffer = createUniformBufferWithView(
    'scatter_add_dynamic_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, hiddenSize, true);
      view.setUint32(8, topK, true);
    },
    recorder
  );

  const bindGroup = await createBindGroupWithValidation(device, {
    label: 'scatter_add_dynamic_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: expertOutputs.buffer } },
      { binding: 2, resource: { buffer: indices } },
      { binding: 3, resource: { buffer: weights } },
      { binding: 4, resource: { buffer: tokenOffsets } },
      { binding: 5, resource: { buffer: outputBuf } },
    ],
  }, `scatter_add_dynamic:${variant}`);

  recorder.getEncoder().clearBuffer(outputBuf);

  const pass = recorder.beginComputePass('scatter_add_dynamic');
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil((numTokens * topK * hiddenSize) / WORKGROUP_SIZES.DEFAULT));
  pass.end();

  return createTensor(outputBuf, expertOutputs.dtype, [numTokens, hiddenSize], 'scatter_add_dynamic_output');
}
