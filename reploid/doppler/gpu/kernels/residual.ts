/**
 * Residual Connection Kernels
 *
 * Provides element-wise addition operations for:
 * - Residual connections (add two tensors)
 * - Bias addition
 */

import { getDevice } from '../device.js';
import { setBufferDtype } from '../buffer-dtypes.js';
import { acquireBuffer } from '../buffer-pool.js';
import type { CommandRecorder } from '../command-recorder.js';
import { WORKGROUP_SIZES } from './constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView } from './utils.js';
import type { OutputBufferOptions } from './types.js';

/** Residual kernel options */
export interface ResidualOptions extends OutputBufferOptions {
  useVec4?: boolean;
  dataOffset?: number;
  biasOffset?: number;
}

/**
 * Run residual add (element-wise addition)
 */
export async function runResidualAdd(
  a: GPUBuffer,
  b: GPUBuffer,
  size: number,
  options: ResidualOptions = {}
): Promise<GPUBuffer> {
  const device = getDevice();
  const { useVec4 = true, outputBuffer = null } = options;

  const variant = useVec4 ? 'vec4' : 'default';
  const pipeline = await createPipeline('residual', variant);

  const outputSize = size * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'residual_output');

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'residual_uniforms',
    16,
    (view) => {
      view.setUint32(0, size, true);
    },
    null,
    device
  );

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'residual_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: a } },
      { binding: 2, resource: { buffer: b } },
      { binding: 3, resource: { buffer: output } },
    ],
  });

  // residual.wgsl: main uses @workgroup_size(256), add_vec4 uses @workgroup_size(64)
  // vec4 variant: 64 threads × 4 elements = 256 elements per workgroup
  const workgroups = useVec4
    ? Math.ceil(size / 256)
    : Math.ceil(size / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'residual');

  uniformBuffer.destroy();

  return output;
}

/**
 * Run bias add
 */
export async function runBiasAdd(
  data: GPUBuffer,
  bias: GPUBuffer,
  numTokens: number,
  dim: number,
  options: ResidualOptions = {}
): Promise<GPUBuffer> {
  const device = getDevice();
  const { dataOffset = 0, biasOffset = 0 } = options;

  const pipeline = await createPipeline('bias_add', 'default');

  // Bias add is in-place, no output buffer creation needed
  const output = data;

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'bias_add_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, dim, true);
      view.setUint32(8, dataOffset, true);
      view.setUint32(12, biasOffset, true);
    },
    null,
    device
  );

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'bias_add_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: data } },
      { binding: 2, resource: { buffer: bias } },
    ],
  });

  const workgroups = Math.ceil((numTokens * dim) / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'bias_add');

  uniformBuffer.destroy();

  return output;
}

/**
 * Record residual add (batched, no submit)
 */
export async function recordResidualAdd(
  recorder: CommandRecorder,
  a: GPUBuffer,
  b: GPUBuffer,
  size: number,
  options: ResidualOptions = {}
): Promise<GPUBuffer> {
  const device = recorder.device;
  const { outputBuffer = null } = options;

  const pipeline = await createPipeline('residual', 'default');

  const outputSize = size * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'residual_output');

  // Uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'residual_uniforms',
    16,
    (view) => {
      view.setUint32(0, size, true);
    },
    recorder
  );

  // Bind group
  const bindGroup = device.createBindGroup({
    label: 'residual_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: a } },
      { binding: 2, resource: { buffer: b } },
      { binding: 3, resource: { buffer: output } },
    ],
  });

  const workgroups = Math.ceil(size / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'residual');

  setBufferDtype(output, 'f32');
  return output;
}

/**
 * Record bias add (batched, no submit)
 */
export async function recordBiasAdd(
  recorder: CommandRecorder,
  data: GPUBuffer,
  bias: GPUBuffer,
  numTokens: number,
  dim: number,
  options: ResidualOptions = {}
): Promise<GPUBuffer> {
  const device = recorder.device;
  const { dataOffset = 0, biasOffset = 0 } = options;

  const pipeline = await createPipeline('bias_add', 'default');

  // Uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'bias_add_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, dim, true);
      view.setUint32(8, dataOffset, true);
      view.setUint32(12, biasOffset, true);
    },
    recorder
  );

  // Bind group
  const bindGroup = device.createBindGroup({
    label: 'bias_add_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: data } },
      { binding: 2, resource: { buffer: bias } },
    ],
  });

  const workgroups = Math.ceil((numTokens * dim) / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'bias_add');

  return data; // In-place operation
}
