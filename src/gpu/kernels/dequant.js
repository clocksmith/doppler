

import { getDevice, getKernelCapabilities } from '../device.js';
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { GPU_LIMITS, TILE_SIZES, WORKGROUP_SIZES } from './constants.js';
import { Q6K_BLOCK_BYTES, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_SIZE } from '../../loader/quantization-constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { getPipelineFast, createUniformBufferWithView, getOrCreateBindGroupLayout } from './utils.js';
import { releaseUniformBuffer } from '../uniform-cache.js';
import { selectRuleValue as selectKernelRuleValue } from './rule-registry.js';
import { selectRuleValue as selectSharedRuleValue } from '../../rules/rule-registry.js';


export function selectDequantKernel(options = {}) {
  const capabilities = getKernelCapabilities();
  const { useVec4 = true, outputDtype = 'f32' } = options;

  const wantsF16Out = outputDtype === 'f16' && capabilities.hasF16;
  return selectKernelRuleValue(
    'dequant',
    'variant',
    { hasSubgroups: capabilities.hasSubgroups, wantsF16Out, useVec4 }
  );
}


function calculateDequantWorkgroups(variant, numBlocks) {
  const QK_K = TILE_SIZES.Q4K_SUPER_BLOCK_SIZE;
  
  let workgroups;

  if (variant.includes('vec4')) {
    workgroups = numBlocks;
  } else if (variant.includes('shared')) {
    workgroups = numBlocks;
  } else {
    workgroups = Math.ceil((numBlocks * QK_K) / (WORKGROUP_SIZES.DEFAULT / 4));
  }

  const maxWorkgroups = GPU_LIMITS.MAX_WORKGROUPS;
  if (workgroups <= maxWorkgroups) {
    return [workgroups, 1, 1];
  }

  const wgY = Math.ceil(workgroups / maxWorkgroups);
  const wgX = Math.min(workgroups, maxWorkgroups);
  return [wgX, wgY, 1];
}


export function createDequantBindGroupLayout() {
  return getOrCreateBindGroupLayout('dequant_bind_group_layout', [
    {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'uniform' },
    },
    {
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'read-only-storage' },
    },
    {
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'storage' },
    },
  ]);
}


export async function dequantize(
  quantized,
  numBlocks,
  options = {}
) {
  const device = getDevice();
  const {
    outputOffset = 0,
    outputBuffer = null,
    outputDtype = 'f32',
  } = options;

  // Select kernel
  const variant = selectDequantKernel({ ...options, outputDtype });
  const pipeline = await getPipelineFast('dequant', variant);

  // Q4_K_M: 256 elements per block
  const QK_K = TILE_SIZES.Q4K_SUPER_BLOCK_SIZE;
  const bytesPerElem = outputDtype === 'f16' ? 2 : 4;
  const outputSize = numBlocks * QK_K * bytesPerElem;

  // Create output buffer if not provided
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'dequant_output');

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'dequant_uniforms',
    16,
    (view) => {
      view.setUint32(0, numBlocks, true);
      view.setUint32(4, outputOffset, true);
      view.setUint32(8, 0, true); // padding
      view.setUint32(12, 0, true); // padding
    },
    null,
    device
  );

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'dequant_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: quantized } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  const workgroups = calculateDequantWorkgroups(variant, numBlocks);
  dispatch(device, pipeline, bindGroup, workgroups, 'dequant');

  // Release uniform buffer back to cache (or destroy if not cached)
  releaseUniformBuffer(uniformBuffer);

  
  const dtype = selectSharedRuleValue('shared', 'dtype', 'f16OrF32FromDtype', { dtype: outputDtype });
  return createTensor(output, dtype, [numBlocks * QK_K], 'dequant_output');
}


// Row-wise dequant is required when K is not aligned to 256; the standard
// dequant output uses padded stride (blocksPerRow * 256), but matmul expects K.
export async function dequantizeRowwise(
  quantized,
  rows,
  K,
  options = {}
) {
  const device = getDevice();
  const { outputBuffer = null, outputDtype = 'f16' } = options;
  const caps = getKernelCapabilities();
  const wantsF16Out = outputDtype === 'f16' && caps.hasF16;
  const finalOutputDtype = wantsF16Out ? 'f16' : 'f32';

  const QK_K = TILE_SIZES.Q4K_SUPER_BLOCK_SIZE;
  const blocksPerRow = Math.ceil(K / QK_K);
  const numBlocks = rows * blocksPerRow;

  const pipeline = await getPipelineFast(
    'dequant',
    wantsF16Out ? 'f16_rowwise' : 'f32_rowwise'
  );

  const bytesPerElem = finalOutputDtype === 'f16' ? 2 : 4;
  const outputSize = rows * K * bytesPerElem;

  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'dequant_rowwise_output');

  const uniformBuffer = createUniformBufferWithView(
    'dequant_rowwise_uniforms',
    16,
    (view) => {
      view.setUint32(0, numBlocks, true);
      view.setUint32(4, blocksPerRow, true);
      view.setUint32(8, K, true);
      view.setUint32(12, rows, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'dequant_rowwise_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: quantized } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  const workgroups = [numBlocks, 1, 1];
  dispatch(device, pipeline, bindGroup, workgroups, 'dequant_rowwise');

  releaseUniformBuffer(uniformBuffer);

  const dtype = selectSharedRuleValue('shared', 'dtype', 'f16OrF32FromDtype', { dtype: finalOutputDtype });
  return createTensor(output, dtype, [rows, K], 'dequant_rowwise_output');
}


export async function dequantizeMXFP4(
  blocks,
  scales,
  totalElements,
  numGroups,
  options = {}
) {
  const device = getDevice();
  const {
    outputBuffer = null,
    groupSize = 32,  // 32 elements per group (16 bytes * 2 nibbles)
  } = options;

  const pipeline = await getPipelineFast('dequant', 'mxfp4');

  const outputSize = totalElements * 4; // F32 output
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'mxfp4_dequant_output');

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'mxfp4_dequant_uniforms',
    16,
    (view) => {
      view.setUint32(0, totalElements, true);
      view.setUint32(4, numGroups, true);
      view.setUint32(8, groupSize, true);
      view.setUint32(12, numGroups * groupSize, true); // row_stride
    },
    null,
    device
  );

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'mxfp4_dequant_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: blocks } },
      { binding: 2, resource: { buffer: scales } },
      { binding: 3, resource: { buffer: output } },
    ],
  });

  const workgroups = Math.ceil(totalElements / WORKGROUP_SIZES.DEFAULT);
  
  const dispatchSize = [
    Math.min(workgroups, GPU_LIMITS.MAX_WORKGROUPS),
    Math.max(1, Math.ceil(workgroups / GPU_LIMITS.MAX_WORKGROUPS)),
    1,
  ];
  dispatch(device, pipeline, bindGroup, dispatchSize, 'mxfp4_dequant');

  releaseUniformBuffer(uniformBuffer);

  return createTensor(output, 'f32', [totalElements], 'mxfp4_dequant_output');
}


export async function dequantizeMXFP4Expert(
  blocks,
  scales,
  expertIdx,
  numExperts,
  outDim,
  numGroups,
  options = {}
) {
  const device = getDevice();
  const { outputBuffer = null, outputDtype = 'f32' } = options;

  const variant = selectKernelRuleValue('dequant', 'mxfp4ExpertVariant', { outputDtype });
  const pipeline = await getPipelineFast('dequant', variant);

  // Output is [out_dim, num_groups * 32] as F32
  const totalOutput = outDim * numGroups * 32;
  const bytesPerElement = outputDtype === 'f16' ? 2 : 4;
  const outputSize = totalOutput * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'mxfp4_expert_output');

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'mxfp4_expert_uniforms',
    32,
    (view) => {
      view.setUint32(0, expertIdx, true);
      view.setUint32(4, numExperts, true);
      view.setUint32(8, outDim, true);
      view.setUint32(12, numGroups, true);
      view.setUint32(16, totalOutput, true);
    },
    null,
    device
  );

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'mxfp4_expert_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: blocks } },
      { binding: 2, resource: { buffer: scales } },
      { binding: 3, resource: { buffer: output } },
    ],
  });

  const workgroups = Math.ceil(totalOutput / WORKGROUP_SIZES.DEFAULT);
  
  const dispatchSize = [
    Math.min(workgroups, GPU_LIMITS.MAX_WORKGROUPS),
    Math.max(1, Math.ceil(workgroups / GPU_LIMITS.MAX_WORKGROUPS)),
    1,
  ];
  dispatch(device, pipeline, bindGroup, dispatchSize, 'mxfp4_expert');

  releaseUniformBuffer(uniformBuffer);

  const dtype = selectSharedRuleValue('shared', 'dtype', 'f16OrF32FromDtype', { dtype: outputDtype });
  return createTensor(output, dtype, [outDim, numGroups * 32], 'mxfp4_expert_output');
}


export async function dequantizeQ6K(
  quantized,
  numBlocks,
  options = {}
) {
  const device = getDevice();
  const {
    outputOffset = 0,
    outputBuffer = null,
    outputDtype = 'f16',  // Q6_K always outputs f16 for now
  } = options;

  // Q6_K only has f16 output kernel currently
  const pipeline = await getPipelineFast('dequant', 'q6k_f16out');

  // Q6_K: 256 elements per block
  const QK_K = TILE_SIZES.Q4K_SUPER_BLOCK_SIZE;
  const bytesPerElem = outputDtype === 'f16' ? 2 : 4;
  const outputSize = numBlocks * QK_K * bytesPerElem;

  // Create output buffer if not provided
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'q6k_dequant_output');

  // Calculate workgroups for 2D dispatch
  const maxWorkgroups = GPU_LIMITS.MAX_WORKGROUPS;
  const workgroupsX = Math.min(numBlocks, maxWorkgroups);

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'q6k_dequant_uniforms',
    16,
    (view) => {
      view.setUint32(0, numBlocks, true);
      view.setUint32(4, outputOffset, true);
      view.setUint32(8, workgroupsX, true); // workgroups_x for 2D dispatch
      view.setUint32(12, 0, true); // padding
    },
    null,
    device
  );

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'q6k_dequant_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: quantized } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  // One workgroup per block, handle 2D dispatch for large counts
  
  const workgroups = [
    workgroupsX,
    numBlocks > maxWorkgroups ? Math.ceil(numBlocks / maxWorkgroups) : 1,
    1
  ];

  dispatch(device, pipeline, bindGroup, workgroups, 'q6k_dequant');

  releaseUniformBuffer(uniformBuffer);

  
  const dtype = selectSharedRuleValue('shared', 'dtype', 'f16OrF32FromDtype', { dtype: outputDtype });
  return createTensor(output, dtype, [numBlocks * QK_K], 'q6k_dequant_output');
}


export async function dequantizeQ8_0(
  quantized,
  numBlocks,
  options = {}
) {
  const device = getDevice();
  const {
    outputOffset = 0,
    outputBuffer = null,
    outputDtype = 'f16',  // Q8_0 outputs f16 for now
  } = options;

  // Q8_0 only has f16 output kernel currently
  const pipeline = await getPipelineFast('dequant', 'q8_0_f16out');

  // Q8_0: 32 elements per block
  const bytesPerElem = outputDtype === 'f16' ? 2 : 4;
  const outputSize = numBlocks * Q8_0_BLOCK_SIZE * bytesPerElem;

  // Create output buffer if not provided
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'q8_0_dequant_output');

  // Calculate workgroups for 2D dispatch
  const maxWorkgroups = GPU_LIMITS.MAX_WORKGROUPS;
  const workgroupsX = Math.min(numBlocks, maxWorkgroups);

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'q8_0_dequant_uniforms',
    16,
    (view) => {
      view.setUint32(0, numBlocks, true);
      view.setUint32(4, outputOffset, true);
      view.setUint32(8, workgroupsX, true); // workgroups_x for 2D dispatch
      view.setUint32(12, 0, true); // padding
    },
    null,
    device
  );

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'q8_0_dequant_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: quantized } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  // One workgroup per block, handle 2D dispatch for large counts
  
  const workgroups = [
    workgroupsX,
    numBlocks > maxWorkgroups ? Math.ceil(numBlocks / maxWorkgroups) : 1,
    1
  ];

  dispatch(device, pipeline, bindGroup, workgroups, 'q8_0_dequant');

  releaseUniformBuffer(uniformBuffer);

  
  const dtype = selectSharedRuleValue('shared', 'dtype', 'f16OrF32FromDtype', { dtype: outputDtype });
  return createTensor(output, dtype, [numBlocks * Q8_0_BLOCK_SIZE], 'q8_0_dequant_output');
}


export async function recordDequantize(
  recorder,
  quantized,
  numBlocks,
  options = {}
) {
  const device = recorder.device;
  const {
    outputOffset = 0,
    outputBuffer = null,
    outputDtype = 'f32',
  } = options;

  // Select kernel
  const variant = selectDequantKernel({ ...options, outputDtype });
  const pipeline = await getPipelineFast('dequant', variant);

  // Q4_K: 256 elements per block
  const QK_K = TILE_SIZES.Q4K_SUPER_BLOCK_SIZE;
  const bytesPerElem = outputDtype === 'f16' ? 2 : 4;
  const outputSize = numBlocks * QK_K * bytesPerElem;

  // Output buffer
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'dequant_output');

  // Uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'dequant_uniforms',
    16,
    (view) => {
      view.setUint32(0, numBlocks, true);
      view.setUint32(4, outputOffset, true);
    },
    recorder
  );

  // Bind group
  const bindGroup = device.createBindGroup({
    label: 'dequant_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: quantized } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  const workgroups = calculateDequantWorkgroups(variant, numBlocks);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'dequant');

  
  const dtype = selectSharedRuleValue('shared', 'dtype', 'f16OrF32FromDtype', { dtype: outputDtype });
  return createTensor(output, dtype, [numBlocks * QK_K], 'dequant_output');
}
