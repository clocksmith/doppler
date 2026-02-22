

import { getDevice, getDeviceEpoch } from '../device.js';
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

function resolveExecution(recorder) {
  return {
    recorder: recorder || null,
    device: recorder?.device || getDevice(),
  };
}

function createMoEUniformBuffer(execution, label, size, writeUniforms) {
  return createUniformBufferWithView(
    label,
    size,
    writeUniforms,
    execution.recorder,
    execution.device
  );
}

function dispatchTopK(execution, pipeline, bindGroup, numTokens) {
  if (execution.recorder) {
    recordDispatch(execution.recorder, pipeline, bindGroup, numTokens, 'topk');
    return;
  }
  dispatch(execution.device, pipeline, bindGroup, numTokens, 'topk');
}

function releaseUniformBuffer(execution, uniformBuffer) {
  if (!execution.recorder) {
    uniformBuffer.destroy();
  }
}

async function executeTopK(recorder, probs, numTokens, numExperts, topK, options = {}) {
  const execution = resolveExecution(recorder);
  const { normalize = true } = options;

  const pipeline = await createPipeline('topk', 'default');

  const indicesSize = numTokens * topK * 4;
  const weightsSize = numTokens * topK * 4;
  const indices = acquireBuffer(indicesSize, undefined, 'topk_indices');
  const weights = acquireBuffer(weightsSize, undefined, 'topk_weights');

  const uniformBuffer = createMoEUniformBuffer(execution, 'topk_uniforms', 16, (view) => {
    view.setUint32(0, numTokens, true);
    view.setUint32(4, numExperts, true);
    view.setUint32(8, topK, true);
    view.setUint32(12, normalize ? 1 : 0, true);
  });

  const bindGroup = execution.device.createBindGroup({
    label: 'topk_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: probs } },
      { binding: 2, resource: { buffer: indices } },
      { binding: 3, resource: { buffer: weights } },
    ],
  });

  dispatchTopK(execution, pipeline, bindGroup, numTokens);
  releaseUniformBuffer(execution, uniformBuffer);

  return { indices, weights };
}

export async function runTopK(probs, numTokens, numExperts, topK, options = {}) {
  return executeTopK(null, probs, numTokens, numExperts, topK, options);
}


export async function recordTopK(recorder, probs, numTokens, numExperts, topK, options = {}) {
  return executeTopK(recorder, probs, numTokens, numExperts, topK, options);
}

// Cached explicit bind group layout for MoE gather (all 6 bindings).
// Internal postmortems cover why this explicit layout is required.

let moeGatherBindGroupLayout = null;
let moeGatherBindGroupLayoutEpoch = -1;


function getMoEGatherBindGroupLayout(device) {
  const epoch = getDeviceEpoch();
  if (moeGatherBindGroupLayout && moeGatherBindGroupLayoutEpoch === epoch) return moeGatherBindGroupLayout;

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
  moeGatherBindGroupLayoutEpoch = epoch;
  return moeGatherBindGroupLayout;
}

// Cached explicit bind group layout for scatter-add dynamic (all 6 bindings)
// Required because auto layout can omit bindings in some driver/compiler paths.
let scatterAddDynamicBindGroupLayout = null;
let scatterAddDynamicBindGroupLayoutEpoch = -1;

function getScatterAddDynamicBindGroupLayout(device) {
  const epoch = getDeviceEpoch();
  if (scatterAddDynamicBindGroupLayout && scatterAddDynamicBindGroupLayoutEpoch === epoch) {
    return scatterAddDynamicBindGroupLayout;
  }

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
  scatterAddDynamicBindGroupLayoutEpoch = epoch;
  return scatterAddDynamicBindGroupLayout;
}

let moeOffsetsBindGroupLayout = null;
let moeOffsetsBindGroupLayoutEpoch = -1;

function getMoEOffsetsBindGroupLayout(device) {
  const epoch = getDeviceEpoch();
  if (moeOffsetsBindGroupLayout && moeOffsetsBindGroupLayoutEpoch === epoch) return moeOffsetsBindGroupLayout;

  moeOffsetsBindGroupLayout = device.createBindGroupLayout({
    label: 'moe_offsets_explicit_layout',
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  moeOffsetsBindGroupLayoutEpoch = epoch;

  return moeOffsetsBindGroupLayout;
}

async function executeMoEGather(recorder, hiddenStates, expertIndices, numTokens, hiddenSize, numExperts, topK, options = {}) {
  const execution = resolveExecution(recorder);
  const { maxTokensPerExpert = numTokens } = options;
  const useF16 = hiddenStates.dtype === 'f16';
  const suffix = selectKernelRuleValue('moe', 'variantSuffix', { useF16 });
  const dtypeLabel = selectSharedRuleValue('shared', 'dtype', 'f16OrF32', { useF16 });

  const explicitLayout = getMoEGatherBindGroupLayout(execution.device);
  const countPipeline = await createPipeline('moe_gather', `count${suffix}`, explicitLayout);
  const gatherPipeline = await createPipeline('moe_gather', `gather${suffix}`, explicitLayout);

  const bytesPerElement = hiddenStates.dtype === 'f16' ? 2 : 4;
  const gatheredSize = numExperts * maxTokensPerExpert * hiddenSize * bytesPerElement;

  const gatherWorkgroupsTotal = Math.ceil((numExperts * maxTokensPerExpert * hiddenSize) / WORKGROUP_SIZES.DEFAULT);
  const gatherDispatch = calculate2DDispatch(gatherWorkgroupsTotal);

  if (!execution.recorder) {
    trace.kernels('moe_gather params', {
      numTokens,
      hiddenSize,
      numExperts,
      topK,
      maxTokensPerExpert,
      gatheredSize,
      gatherWorkgroups: gatherWorkgroupsTotal,
      gatherDispatch,
    });
  }
  const tokenCountsSize = numExperts * 4;
  const tokenMapSize = numExperts * maxTokensPerExpert * 2 * 4;

  const gatheredBuffer = acquireBuffer(gatheredSize, undefined, 'moe_gathered');
  const tokenCounts = acquireBuffer(tokenCountsSize, undefined, 'moe_token_counts');
  const tokenMap = acquireBuffer(tokenMapSize, undefined, 'moe_token_map');

  const uniformBuffer = createMoEUniformBuffer(execution, 'moe_gather_uniforms', 32, (view) => {
    view.setUint32(0, numTokens, true);
    view.setUint32(4, hiddenSize, true);
    view.setUint32(8, numExperts, true);
    view.setUint32(12, topK, true);
    view.setUint32(16, maxTokensPerExpert, true);
    view.setUint32(20, gatherDispatch.threadsPerRow, true);
    view.setUint32(24, 0, true);
    view.setUint32(28, 0, true);
  });

  const bindGroup = await createBindGroupWithValidation(execution.device, {
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

  const countWorkgroups = Math.ceil((numTokens * topK) / WORKGROUP_SIZES.DEFAULT);

  if (execution.recorder) {
    const encoder = execution.recorder.getEncoder();
    encoder.clearBuffer(tokenCounts);

    const countPass = execution.recorder.beginComputePass('moe_gather_count');
    countPass.setPipeline(countPipeline);
    countPass.setBindGroup(0, bindGroup);
    countPass.dispatchWorkgroups(countWorkgroups);
    countPass.end();

    const gatherPass = execution.recorder.beginComputePass('moe_gather_gather');
    gatherPass.setPipeline(gatherPipeline);
    gatherPass.setBindGroup(0, bindGroup);
    gatherPass.dispatchWorkgroups(gatherDispatch.x, gatherDispatch.y, 1);
    gatherPass.end();
  } else {
    const encoder = execution.device.createCommandEncoder({ label: 'moe_gather_encoder' });
    encoder.clearBuffer(tokenCounts);

    const countPass = encoder.beginComputePass({ label: 'moe_gather_count_pass' });
    countPass.setPipeline(countPipeline);
    countPass.setBindGroup(0, bindGroup);
    countPass.dispatchWorkgroups(countWorkgroups);
    countPass.end();

    const gatherPass = encoder.beginComputePass({ label: 'moe_gather_gather_pass' });
    gatherPass.setPipeline(gatherPipeline);
    gatherPass.setBindGroup(0, bindGroup);
    gatherPass.dispatchWorkgroups(gatherDispatch.x, gatherDispatch.y, 1);
    gatherPass.end();

    execution.device.queue.submit([encoder.finish()]);
  }

  releaseUniformBuffer(execution, uniformBuffer);

  const gathered = createTensor(
    gatheredBuffer,
    hiddenStates.dtype,
    [numExperts, maxTokensPerExpert, hiddenSize],
    'moe_gathered'
  );

  return { gathered, tokenCounts, tokenMap, maxTokensPerExpert };
}

export async function runMoEGather(hiddenStates, expertIndices, numTokens, hiddenSize, numExperts, topK, options = {}) {
  return executeMoEGather(null, hiddenStates, expertIndices, numTokens, hiddenSize, numExperts, topK, options);
}


export async function recordMoEGather(recorder, hiddenStates, expertIndices, numTokens, hiddenSize, numExperts, topK, options = {}) {
  return executeMoEGather(recorder, hiddenStates, expertIndices, numTokens, hiddenSize, numExperts, topK, options);
}

async function executeScatterAdd(recorder, expertOutputs, indices, weights, numTokens, hiddenSize, numExperts, topK, options = {}) {
  const execution = resolveExecution(recorder);
  const { outputBuffer = null } = options;

  const pipeline = await createPipeline('scatter_add', 'default');

  const bytesPerElement = expertOutputs.dtype === 'f16' ? 2 : 4;
  const outputSize = numTokens * hiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'scatter_add_output');

  const uniformBuffer = createMoEUniformBuffer(execution, 'scatter_add_uniforms', 16, (view) => {
    view.setUint32(0, numTokens, true);
    view.setUint32(4, hiddenSize, true);
    view.setUint32(8, topK, true);
    view.setUint32(12, numExperts, true);
  });

  const bindGroup = execution.device.createBindGroup({
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

  const workgroups = Math.ceil((numTokens * hiddenSize) / WORKGROUP_SIZES.DEFAULT);

  if (execution.recorder) {
    execution.recorder.getEncoder().clearBuffer(outputBuf);

    const pass = execution.recorder.beginComputePass('scatter_add');
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups);
    pass.end();
  } else {
    const encoder = execution.device.createCommandEncoder({ label: 'scatter_add_encoder' });
    encoder.clearBuffer(outputBuf);
    const pass = encoder.beginComputePass({ label: 'scatter_add_pass' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups);
    pass.end();
    execution.device.queue.submit([encoder.finish()]);
  }

  releaseUniformBuffer(execution, uniformBuffer);

  return createTensor(outputBuf, expertOutputs.dtype, [numTokens, hiddenSize], 'scatter_add_output');
}

export async function runScatterAdd(expertOutputs, indices, weights, numTokens, hiddenSize, numExperts, topK, options = {}) {
  return executeScatterAdd(null, expertOutputs, indices, weights, numTokens, hiddenSize, numExperts, topK, options);
}


export async function recordScatterAdd(recorder, expertOutputs, indices, weights, numTokens, hiddenSize, numExperts, topK, options = {}) {
  return executeScatterAdd(recorder, expertOutputs, indices, weights, numTokens, hiddenSize, numExperts, topK, options);
}

async function executeMoEBuildTokenOffsets(recorder, tokenCounts, tokenMap, numTokens, numExperts, topK, maxTokensPerExpert, options = {}) {
  const execution = resolveExecution(recorder);
  const { outputBuffer = null } = options;

  const explicitLayout = getMoEOffsetsBindGroupLayout(execution.device);
  const pipeline = await createPipeline('moe_offsets', 'default', explicitLayout);

  const tokenOffsetsSize = numTokens * topK * 4;
  const tokenOffsets = outputBuffer || acquireBuffer(tokenOffsetsSize, undefined, 'moe_token_offsets');

  const uniformBuffer = createMoEUniformBuffer(execution, 'moe_offsets_uniforms', 32, (view) => {
    view.setUint32(0, numTokens, true);
    view.setUint32(4, numExperts, true);
    view.setUint32(8, topK, true);
    view.setUint32(12, maxTokensPerExpert, true);
    view.setUint32(16, 0, true);
    view.setUint32(20, 0, true);
    view.setUint32(24, 0, true);
    view.setUint32(28, 0, true);
  });

  const bindGroup = await createBindGroupWithValidation(execution.device, {
    label: 'moe_offsets_bind_group',
    layout: explicitLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: tokenCounts } },
      { binding: 2, resource: { buffer: tokenMap } },
      { binding: 3, resource: { buffer: tokenOffsets } },
    ],
  }, 'moe_offsets');

  const totalSlots = numExperts * maxTokensPerExpert;
  const workgroups = Math.ceil(totalSlots / WORKGROUP_SIZES.DEFAULT);

  if (execution.recorder) {
    const pass = execution.recorder.beginComputePass('moe_offsets');
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups);
    pass.end();
  } else {
    const encoder = execution.device.createCommandEncoder({ label: 'moe_offsets_encoder' });
    const pass = encoder.beginComputePass({ label: 'moe_offsets_pass' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups);
    pass.end();
    execution.device.queue.submit([encoder.finish()]);
  }

  releaseUniformBuffer(execution, uniformBuffer);
  return tokenOffsets;
}

export async function runMoEBuildTokenOffsets(tokenCounts, tokenMap, numTokens, numExperts, topK, maxTokensPerExpert, options = {}) {
  return executeMoEBuildTokenOffsets(null, tokenCounts, tokenMap, numTokens, numExperts, topK, maxTokensPerExpert, options);
}

export async function recordMoEBuildTokenOffsets(recorder, tokenCounts, tokenMap, numTokens, numExperts, topK, maxTokensPerExpert, options = {}) {
  return executeMoEBuildTokenOffsets(recorder, tokenCounts, tokenMap, numTokens, numExperts, topK, maxTokensPerExpert, options);
}

async function executeScatterAddDynamic(recorder, expertOutputs, indices, weights, tokenOffsets, numTokens, hiddenSize, topK, options = {}) {
  const execution = resolveExecution(recorder);
  const { outputBuffer = null, weightsDtype = 'f32' } = options;

  if (weightsDtype === 'f16' && expertOutputs.dtype !== 'f16') {
    throw new Error('ScatterAddDynamic f16 weights require f16 expert outputs');
  }

  const variant = selectKernelRuleValue('moe', 'scatterAddVariant', {
    outputDtype: expertOutputs.dtype,
    weightsDtype,
  });
  const explicitLayout = getScatterAddDynamicBindGroupLayout(execution.device);
  const pipeline = await createPipeline('scatter_add', variant, explicitLayout);

  const bytesPerElement = expertOutputs.dtype === 'f16' ? 2 : 4;
  const outputSize = numTokens * hiddenSize * bytesPerElement;
  const outputBuf = outputBuffer || acquireBuffer(outputSize, undefined, 'scatter_add_dynamic_output');

  const uniformBuffer = createMoEUniformBuffer(execution, 'scatter_add_dynamic_uniforms', 16, (view) => {
    view.setUint32(0, numTokens, true);
    view.setUint32(4, hiddenSize, true);
    view.setUint32(8, topK, true);
  });

  const bindGroup = await createBindGroupWithValidation(execution.device, {
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

  const workgroups = Math.ceil((numTokens * topK * hiddenSize) / WORKGROUP_SIZES.DEFAULT);

  if (execution.recorder) {
    execution.recorder.getEncoder().clearBuffer(outputBuf);

    const pass = execution.recorder.beginComputePass('scatter_add_dynamic');
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups);
    pass.end();
  } else {
    const encoder = execution.device.createCommandEncoder({ label: 'scatter_add_dynamic_encoder' });
    encoder.clearBuffer(outputBuf);
    const pass = encoder.beginComputePass({ label: 'scatter_add_dynamic_pass' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups);
    pass.end();
    execution.device.queue.submit([encoder.finish()]);
  }

  releaseUniformBuffer(execution, uniformBuffer);

  return createTensor(outputBuf, expertOutputs.dtype, [numTokens, hiddenSize], 'scatter_add_dynamic_output');
}

export async function runScatterAddDynamic(expertOutputs, indices, weights, tokenOffsets, numTokens, hiddenSize, topK, options = {}) {
  return executeScatterAddDynamic(null, expertOutputs, indices, weights, tokenOffsets, numTokens, hiddenSize, topK, options);
}


export async function recordScatterAddDynamic(recorder, expertOutputs, indices, weights, tokenOffsets, numTokens, hiddenSize, topK, options = {}) {
  return executeScatterAddDynamic(recorder, expertOutputs, indices, weights, tokenOffsets, numTokens, hiddenSize, topK, options);
}
