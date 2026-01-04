/**
 * Mixture of Experts (MoE) Kernels
 *
 * Provides kernels for MoE routing and token distribution:
 * - Top-K expert selection
 * - MoE token gathering (dispatching tokens to experts)
 * - Scatter-add (collecting expert outputs back to tokens)
 */

import { getDevice } from '../device.js';
import { acquireBuffer } from '../buffer-pool.js';
import { createTensor, type Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import { WORKGROUP_SIZES } from './constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView } from './utils.js';
import type { OutputBufferOptions } from './types.js';

/** MoE kernel options */
export interface MoEOptions extends OutputBufferOptions {
  normalize?: boolean;
  maxTokensPerExpert?: number;
}

/** MoE gather result */
export interface MoEGatherResult {
  gathered: Tensor;
  tokenCounts: GPUBuffer;
  tokenMap: GPUBuffer;
  maxTokensPerExpert: number;
}

/**
 * Run top-K expert selection
 */
export async function runTopK(
  probs: GPUBuffer,
  numTokens: number,
  numExperts: number,
  topK: number,
  options: MoEOptions = {}
): Promise<{ indices: GPUBuffer; weights: GPUBuffer }> {
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

/**
 * Record top-K expert selection (batched, no submit)
 */
export async function recordTopK(
  recorder: CommandRecorder,
  probs: GPUBuffer,
  numTokens: number,
  numExperts: number,
  topK: number,
  options: MoEOptions = {}
): Promise<{ indices: GPUBuffer; weights: GPUBuffer }> {
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

// Cached explicit bind group layout for MoE gather (all 6 bindings)
// See docs/postmortems/MOE-EXPLICIT-LAYOUT-POSTMORTEM.md for why this is needed
let moeGatherBindGroupLayout: GPUBindGroupLayout | null = null;

function getMoEGatherBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
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

/**
 * Run MoE gather (dispatch tokens to experts)
 * Returns gathered hidden states organized by expert, along with token counts and mapping
 */
export async function runMoEGather(
  hiddenStates: Tensor,
  expertIndices: GPUBuffer,
  numTokens: number,
  hiddenSize: number,
  numExperts: number,
  topK: number,
  options: MoEOptions = {}
): Promise<MoEGatherResult> {
  const device = getDevice();
  const { maxTokensPerExpert = numTokens } = options;

  // Use explicit bind group layout (required because count_and_map doesn't use all bindings)
  const explicitLayout = getMoEGatherBindGroupLayout(device);

  // Two-phase approach: count_and_map builds token assignments, gather copies hidden states
  const countPipeline = await createPipeline('moe_gather', 'count', explicitLayout);
  const gatherPipeline = await createPipeline('moe_gather', 'gather', explicitLayout);

  // Output buffers per WGSL shader:
  // - gathered: [numExperts, maxTokensPerExpert, hiddenSize]
  // - tokenCounts: [numExperts]
  // - tokenMap: [numExperts, maxTokensPerExpert, 2] (tokenIdx, kIdx)
  const bytesPerElement = hiddenStates.dtype === 'f16' ? 2 : 4;
  const gatheredSize = numExperts * maxTokensPerExpert * hiddenSize * bytesPerElement;
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
      view.setUint32(20, 0, true); // _pad1
      view.setUint32(24, 0, true); // _pad2
      view.setUint32(28, 0, true); // _pad3
    },
    null,
    device
  );

  // Create bind group with explicit layout (all 6 bindings)
  const bindGroup = device.createBindGroup({
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
  });

  // Phase 1: Count tokens per expert and build token map
  const encoder = device.createCommandEncoder({ label: 'moe_gather_encoder' });
  encoder.clearBuffer(tokenCounts); // Zero-initialize tokenCounts (atomics start at 0)

  const countPass = encoder.beginComputePass({ label: 'moe_gather_count_pass' });
  countPass.setPipeline(countPipeline);
  countPass.setBindGroup(0, bindGroup);
  const countWorkgroups = Math.ceil((numTokens * topK) / WORKGROUP_SIZES.DEFAULT);
  countPass.dispatchWorkgroups(countWorkgroups);
  countPass.end();

  // Phase 2: Gather hidden states based on token map
  const gatherPass = encoder.beginComputePass({ label: 'moe_gather_gather_pass' });
  gatherPass.setPipeline(gatherPipeline);
  gatherPass.setBindGroup(0, bindGroup);
  const gatherWorkgroups = Math.ceil((numExperts * maxTokensPerExpert * hiddenSize) / WORKGROUP_SIZES.DEFAULT);
  gatherPass.dispatchWorkgroups(gatherWorkgroups);
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

/**
 * Record MoE gather (batched, no submit)
 */
export async function recordMoEGather(
  recorder: CommandRecorder,
  hiddenStates: Tensor,
  expertIndices: GPUBuffer,
  numTokens: number,
  hiddenSize: number,
  numExperts: number,
  topK: number,
  options: MoEOptions = {}
): Promise<MoEGatherResult> {
  const device = recorder.device;
  const { maxTokensPerExpert = numTokens } = options;

  // Use explicit bind group layout (required because count_and_map doesn't use all bindings)
  const explicitLayout = getMoEGatherBindGroupLayout(device);

  // Two-phase approach: count_and_map builds token assignments, gather copies hidden states
  const countPipeline = await createPipeline('moe_gather', 'count', explicitLayout);
  const gatherPipeline = await createPipeline('moe_gather', 'gather', explicitLayout);

  const bytesPerElement = hiddenStates.dtype === 'f16' ? 2 : 4;
  const gatheredSize = numExperts * maxTokensPerExpert * hiddenSize * bytesPerElement;
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
      view.setUint32(20, 0, true); // _pad1
      view.setUint32(24, 0, true); // _pad2
      view.setUint32(28, 0, true); // _pad3
    },
    recorder
  );

  // Create bind group with explicit layout (all 6 bindings)
  const bindGroup = device.createBindGroup({
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
  });

  const encoder = recorder.getEncoder();
  encoder.clearBuffer(tokenCounts);

  // Phase 1: Count tokens per expert and build token map
  const countPass = recorder.beginComputePass('moe_gather_count');
  countPass.setPipeline(countPipeline);
  countPass.setBindGroup(0, bindGroup);
  countPass.dispatchWorkgroups(Math.ceil((numTokens * topK) / WORKGROUP_SIZES.DEFAULT));
  countPass.end();

  // Phase 2: Gather hidden states based on token map
  const gatherPass = recorder.beginComputePass('moe_gather_gather');
  gatherPass.setPipeline(gatherPipeline);
  gatherPass.setBindGroup(0, bindGroup);
  gatherPass.dispatchWorkgroups(Math.ceil((numExperts * maxTokensPerExpert * hiddenSize) / WORKGROUP_SIZES.DEFAULT));
  gatherPass.end();

  const gathered = createTensor(
    gatheredBuffer,
    hiddenStates.dtype,
    [numExperts, maxTokensPerExpert, hiddenSize],
    'moe_gathered'
  );

  return { gathered, tokenCounts, tokenMap, maxTokensPerExpert };
}

/**
 * Run scatter-add (collect expert outputs back to tokens)
 */
export async function runScatterAdd(
  expertOutputs: Tensor,
  indices: GPUBuffer,
  weights: GPUBuffer,
  numTokens: number,
  hiddenSize: number,
  numExperts: number,
  topK: number,
  options: MoEOptions = {}
): Promise<Tensor> {
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

/**
 * Record scatter-add (batched, no submit)
 */
export async function recordScatterAdd(
  recorder: CommandRecorder,
  expertOutputs: Tensor,
  indices: GPUBuffer,
  weights: GPUBuffer,
  numTokens: number,
  hiddenSize: number,
  numExperts: number,
  topK: number,
  options: MoEOptions = {}
): Promise<Tensor> {
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

/**
 * Run dynamic scatter-add with token offsets
 */
export async function runScatterAddDynamic(
  expertOutputs: Tensor,
  indices: GPUBuffer,
  weights: GPUBuffer,
  tokenOffsets: GPUBuffer,
  numTokens: number,
  hiddenSize: number,
  topK: number,
  options: MoEOptions = {}
): Promise<Tensor> {
  const device = getDevice();
  const { outputBuffer = null } = options;

  const pipeline = await createPipeline('scatter_add', 'dynamic');

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
  const bindGroup = device.createBindGroup({
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
  });

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

/**
 * Record dynamic scatter-add (batched, no submit)
 */
export async function recordScatterAddDynamic(
  recorder: CommandRecorder,
  expertOutputs: Tensor,
  indices: GPUBuffer,
  weights: GPUBuffer,
  tokenOffsets: GPUBuffer,
  numTokens: number,
  hiddenSize: number,
  topK: number,
  options: MoEOptions = {}
): Promise<Tensor> {
  const device = recorder.device;
  const { outputBuffer = null } = options;

  const pipeline = await createPipeline('scatter_add', 'dynamic');
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

  const bindGroup = device.createBindGroup({
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
  });

  recorder.getEncoder().clearBuffer(outputBuf);

  const pass = recorder.beginComputePass('scatter_add_dynamic');
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil((numTokens * topK * hiddenSize) / WORKGROUP_SIZES.DEFAULT));
  pass.end();

  return createTensor(outputBuf, expertOutputs.dtype, [numTokens, hiddenSize], 'scatter_add_dynamic_output');
}
