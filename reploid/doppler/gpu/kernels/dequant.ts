/**
 * Dequantization Kernels
 *
 * Provides dequantization operations for:
 * - Q4_K_M quantization (GGUF format)
 * - MXFP4 quantization (GPT-OSS format)
 * - F16/F32 output support
 * - Subgroup and shared memory variants
 */

import { getDevice, getKernelCapabilities } from '../device.js';
import { setBufferDtype } from '../buffer-dtypes.js';
import { acquireBuffer } from '../buffer-pool.js';
import type { CommandRecorder } from '../command-recorder.js';
import { GPU_LIMITS, TILE_SIZES, WORKGROUP_SIZES } from './constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { getPipelineFast, createUniformBufferWithView, getOrCreateBindGroupLayout } from './utils.js';
import { releaseUniformBuffer } from '../uniform-cache.js';
import type { OutputBufferOptions, OutputDtypeOptions, OutputOffsetOptions, Vec4Options } from './types.js';

/** Dequantization kernel options */
export interface DequantOptions extends OutputBufferOptions, OutputOffsetOptions, OutputDtypeOptions, Vec4Options {
  groupSize?: number;
}

/**
 * Select the best dequantization kernel variant
 */
export function selectDequantKernel(options: DequantOptions = {}): string {
  const capabilities = getKernelCapabilities();
  const { useVec4 = true, outputDtype = 'f32' } = options;

  const wantsF16Out = outputDtype === 'f16' && capabilities.hasF16;

  if (capabilities.hasSubgroups) {
    if (wantsF16Out) {
      return useVec4 ? 'subgroup_vec4_f16out' : 'subgroup_f16out';
    }
    return useVec4 ? 'subgroup_vec4' : 'subgroup';
  }

  if (wantsF16Out) {
    return useVec4 ? 'shared_vec4_f16out' : 'shared_f16out';
  }

  return useVec4 ? 'shared_vec4' : 'shared';
}

function calculateDequantWorkgroups(variant: string, numBlocks: number): [number, number, number] {
  const QK_K = TILE_SIZES.Q4K_SUPER_BLOCK_SIZE;
  let workgroups: number;

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

/**
 * Create bind group layout for dequant operation
 */
export function createDequantBindGroupLayout(): GPUBindGroupLayout {
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

/**
 * Run Q4_K_M dequantization
 */
export async function dequantize(
  quantized: GPUBuffer,
  numBlocks: number,
  options: DequantOptions = {}
): Promise<GPUBuffer> {
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
  setBufferDtype(output, outputDtype === 'f16' ? 'f16' : 'f32');

  return output;
}

/**
 * Dequantize MXFP4 weights (GPT-OSS format)
 */
export async function dequantizeMXFP4(
  blocks: GPUBuffer,
  scales: GPUBuffer,
  totalElements: number,
  numGroups: number,
  options: DequantOptions = {}
): Promise<GPUBuffer> {
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
  const dispatchSize: [number, number, number] = [
    Math.min(workgroups, GPU_LIMITS.MAX_WORKGROUPS),
    Math.max(1, Math.ceil(workgroups / GPU_LIMITS.MAX_WORKGROUPS)),
    1,
  ];
  dispatch(device, pipeline, bindGroup, dispatchSize, 'mxfp4_dequant');

  releaseUniformBuffer(uniformBuffer);
  setBufferDtype(output, 'f32');

  return output;
}

/**
 * Dequantize MXFP4 expert weights (extracts single expert from packed tensor)
 */
export async function dequantizeMXFP4Expert(
  blocks: GPUBuffer,
  scales: GPUBuffer,
  expertIdx: number,
  numExperts: number,
  outDim: number,
  numGroups: number,
  options: DequantOptions = {}
): Promise<GPUBuffer> {
  const device = getDevice();
  const { outputBuffer = null } = options;

  const pipeline = await getPipelineFast('dequant', 'mxfp4_expert');

  // Output is [out_dim, num_groups * 32] as F32
  const totalOutput = outDim * numGroups * 32;
  const outputSize = totalOutput * 4;
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
  const dispatchSize: [number, number, number] = [
    Math.min(workgroups, GPU_LIMITS.MAX_WORKGROUPS),
    Math.max(1, Math.ceil(workgroups / GPU_LIMITS.MAX_WORKGROUPS)),
    1,
  ];
  dispatch(device, pipeline, bindGroup, dispatchSize, 'mxfp4_expert');

  releaseUniformBuffer(uniformBuffer);
  setBufferDtype(output, 'f32');

  return output;
}

/**
 * Q6_K block size in bytes (210 bytes per 256 elements)
 */
const Q6K_BLOCK_BYTES = 210;

/**
 * Run Q6_K dequantization
 *
 * Q6_K format: 210 bytes per 256 elements
 * - d: 2 bytes (f16 super-block scale)
 * - ql: 128 bytes (low 4 bits packed)
 * - qh: 64 bytes (high 2 bits packed)
 * - scales: 16 bytes (8-bit signed block scales)
 */
export async function dequantizeQ6K(
  quantized: GPUBuffer,
  numBlocks: number,
  options: DequantOptions = {}
): Promise<GPUBuffer> {
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
  const workgroups: [number, number, number] = [
    workgroupsX,
    numBlocks > maxWorkgroups ? Math.ceil(numBlocks / maxWorkgroups) : 1,
    1
  ];

  dispatch(device, pipeline, bindGroup, workgroups, 'q6k_dequant');

  releaseUniformBuffer(uniformBuffer);
  setBufferDtype(output, outputDtype === 'f16' ? 'f16' : 'f32');

  return output;
}

/**
 * Q8_0 block size in bytes (34 bytes per 32 elements)
 */
const Q8_0_BLOCK_BYTES = 34;

/**
 * Q8_0 block size in elements
 */
const Q8_0_BLOCK_SIZE = 32;

/**
 * Run Q8_0 dequantization
 *
 * Q8_0 format: 34 bytes per 32 elements
 * - d: 2 bytes (f16 scale)
 * - qs: 32 bytes (int8 quantized values)
 *
 * Dequant: output[i] = d * qs[i]
 */
export async function dequantizeQ8_0(
  quantized: GPUBuffer,
  numBlocks: number,
  options: DequantOptions = {}
): Promise<GPUBuffer> {
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
  const workgroups: [number, number, number] = [
    workgroupsX,
    numBlocks > maxWorkgroups ? Math.ceil(numBlocks / maxWorkgroups) : 1,
    1
  ];

  dispatch(device, pipeline, bindGroup, workgroups, 'q8_0_dequant');

  releaseUniformBuffer(uniformBuffer);
  setBufferDtype(output, outputDtype === 'f16' ? 'f16' : 'f32');

  return output;
}

/**
 * Record Q4_K_M dequantization (batched, no submit)
 */
export async function recordDequantize(
  recorder: CommandRecorder,
  quantized: GPUBuffer,
  numBlocks: number,
  options: DequantOptions = {}
): Promise<GPUBuffer> {
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

  setBufferDtype(output, outputDtype === 'f16' ? 'f16' : 'f32');
  return output;
}
