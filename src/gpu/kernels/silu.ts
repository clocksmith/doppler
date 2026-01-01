/**
 * SiLU (Swish) Activation Kernels
 *
 * Provides SiLU activation with variants:
 * - Standard SiLU: x * sigmoid(x)
 * - SiLU with gating (for GLU layers)
 * - SwiGLU with row-split bias
 */

import { getDevice } from '../device.js';
import { setBufferDtype } from '../buffer-dtypes.js';
import { acquireBuffer } from '../buffer-pool.js';
import type { CommandRecorder } from '../command-recorder.js';
import { WORKGROUP_SIZES } from './constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { getPipelineFast, createUniformBufferWithView } from './utils.js';
import type { OutputBufferOptions } from './types.js';

/** SiLU kernel options */
export interface SiLUOptions extends OutputBufferOptions {
  size?: number | null;
  gate?: GPUBuffer | null;
  useVec4?: boolean;
  biasOffset?: number;
}

/**
 * Run SiLU activation
 */
export async function runSiLU(
  input: GPUBuffer,
  options: SiLUOptions = {}
): Promise<GPUBuffer> {
  const device = getDevice();
  const { size, gate = null, outputBuffer = null, useVec4 = false } = options;

  const variant = gate ? 'gate' : (useVec4 ? 'vec4' : 'default');
  const pipeline = await getPipelineFast('silu', variant);

  const inferredSize = size || (input.size / 4);
  const outputSize = inferredSize * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'silu_output');

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'silu_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredSize, true);
    },
    null,
    device
  );

  // Create bind group
  // WGSL bindings: 0=uniforms, 1=input, 2=output, 3=gate, 4=bias
  const entries: GPUBindGroupEntry[] = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: input } },
    { binding: 2, resource: { buffer: output } },
  ];
  if (gate) {
    entries.push({ binding: 3, resource: { buffer: gate } });
  }

  const bindGroup = device.createBindGroup({
    label: 'silu_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  const workgroups = Math.ceil(inferredSize / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'silu');

  uniformBuffer.destroy();

  return output;
}

/**
 * Run SwiGLU with row-split bias
 */
export async function runSwiGLURowsplitBias(
  input: GPUBuffer,
  bias: GPUBuffer,
  numTokens: number,
  dim: number,
  options: SiLUOptions = {}
): Promise<GPUBuffer> {
  const device = getDevice();
  const { outputBuffer = null, biasOffset = 0 } = options;

  const pipeline = await getPipelineFast('swiglu', 'rowsplit_bias');

  const outputSize = numTokens * dim * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'swiglu_output');

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'swiglu_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens, true);
      view.setUint32(4, dim, true);
      view.setUint32(8, biasOffset, true);
    },
    null,
    device
  );

  // Create bind group
  const bindGroup = device.createBindGroup({
    label: 'swiglu_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: bias } },
      { binding: 3, resource: { buffer: output } },
    ],
  });

  const workgroups = Math.ceil((numTokens * dim) / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'swiglu');

  uniformBuffer.destroy();

  return output;
}

/**
 * Row-split SiLU options for fused gate+up FFN
 */
export interface SiLURowSplitOptions extends OutputBufferOptions {
  numTokens: number;
  dim: number;  // intermediateSize
  activation?: 'silu' | 'gelu';
}

/**
 * Run row-split SiLU/GELU for fused gate+up FFN.
 *
 * Input: [numTokens, 2*dim] where each row is [gate[0..dim), up[0..dim)]
 * Output: [numTokens, dim] = activation(gate) * up
 */
export async function runSiLURowSplit(
  input: GPUBuffer,
  options: SiLURowSplitOptions
): Promise<GPUBuffer> {
  const device = getDevice();
  const { numTokens, dim, activation = 'silu', outputBuffer = null } = options;

  const variant = activation === 'gelu' ? 'geglu_rowsplit' : 'gate_rowsplit';
  const pipeline = await getPipelineFast('silu', variant);

  const outputSize = numTokens * dim * 4;  // f32 elements
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'silu_rowsplit_output');

  // Create uniform buffer
  // size = output elements, hasBias = dim (repurposed for rowsplit)
  const uniformBuffer = createUniformBufferWithView(
    'silu_rowsplit_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens * dim, true);  // size
      view.setUint32(4, dim, true);               // hasBias = dim
      view.setUint32(8, 0, true);                 // hasGate (unused)
    },
    null,
    device
  );

  // Create bind group - rowsplit only needs uniforms, input, and output (no gate binding)
  const bindGroup = device.createBindGroup({
    label: 'silu_rowsplit_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  const workgroups = Math.ceil((numTokens * dim) / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'silu_rowsplit');

  uniformBuffer.destroy();
  setBufferDtype(output, 'f32');

  return output;
}

/**
 * Record row-split SiLU/GELU (batched, no submit)
 */
export async function recordSiLURowSplit(
  recorder: CommandRecorder,
  input: GPUBuffer,
  options: SiLURowSplitOptions
): Promise<GPUBuffer> {
  const device = recorder.device;
  const { numTokens, dim, activation = 'silu', outputBuffer = null } = options;

  const variant = activation === 'gelu' ? 'geglu_rowsplit' : 'gate_rowsplit';
  const pipeline = await getPipelineFast('silu', variant);

  const outputSize = numTokens * dim * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'silu_rowsplit_output');

  // Uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'silu_rowsplit_uniforms',
    16,
    (view) => {
      view.setUint32(0, numTokens * dim, true);  // size
      view.setUint32(4, dim, true);               // hasBias = dim
    },
    recorder
  );

  // Rowsplit only needs uniforms, input, and output (no gate binding)
  const bindGroup = device.createBindGroup({
    label: 'silu_rowsplit_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  const workgroups = Math.ceil((numTokens * dim) / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'silu_rowsplit');

  setBufferDtype(output, 'f32');
  return output;
}

/**
 * Record SiLU (batched, no submit)
 * Supports gated variant when options.gate is provided.
 */
export async function recordSiLU(
  recorder: CommandRecorder,
  input: GPUBuffer,
  options: SiLUOptions = {}
): Promise<GPUBuffer> {
  const device = recorder.device;
  const { size, gate = null, outputBuffer = null } = options;

  const variant = gate ? 'gate' : 'default';
  const pipeline = await getPipelineFast('silu', variant);

  const inferredSize = size || (input.size / 4);
  const outputSize = inferredSize * 4;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'silu_output');

  // Uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'silu_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredSize, true);
    },
    recorder
  );

  // Bind group entries - gate variant needs binding 3
  const gateBuffer = gate || input; // Use input as dummy if no gate
  const entries: GPUBindGroupEntry[] = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: input } },
    { binding: 2, resource: { buffer: output } },
  ];

  // Add gate binding for gate variant
  if (gate) {
    entries.push({ binding: 3, resource: { buffer: gateBuffer } });
  }

  const bindGroup = device.createBindGroup({
    label: 'silu_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  const workgroups = Math.ceil(inferredSize / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'silu');

  setBufferDtype(output, 'f32');
  return output;
}
