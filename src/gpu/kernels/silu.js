/**
 * SiLU (Swish) Activation Kernels
 *
 * Provides SiLU activation with variants:
 * - Standard SiLU: x * sigmoid(x)
 * - SiLU with gating (for GLU layers)
 * - SwiGLU with row-split bias
 */

import { getDevice } from '../device.js';
import { acquireBuffer } from '../buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { getPipelineFast, createUniformBufferWithView } from './utils.js';

// =============================================================================
// SiLU Variant Lookup
// =============================================================================

/**
 * SiLU variant lookup table keyed by "${base}/${f16}".
 * Replaces string concatenation for F16 variant selection.
 * @type {Record<string, string>}
 */
const SILU_VARIANTS = {
  'default/false': 'default',
  'default/true': 'default_f16',
  'vec4/false': 'vec4',
  'vec4/true': 'vec4_f16',
  'gate/false': 'gate',
  'gate/true': 'gate_f16',
  'gate_rowsplit/false': 'gate_rowsplit',
  'gate_rowsplit/true': 'gate_rowsplit_f16',
  'geglu_rowsplit/false': 'geglu_rowsplit',
  'geglu_rowsplit/true': 'geglu_rowsplit_f16',
};

/**
 * Select SiLU variant based on base variant and F16 mode.
 * @param {string} base
 * @param {boolean} isF16
 * @returns {string}
 */
function selectSiLUVariant(base, isF16) {
  const key = `${base}/${isF16}`;
  return SILU_VARIANTS[key] ?? base;
}

/**
 * Check if F16 can be used based on tensor dtype.
 * @param {import('../tensor.js').Tensor} input
 * @returns {boolean}
 */
function canUseF16(input) {
  return input.dtype === 'f16';
}

/**
 * Create bind group entries for SiLU, optionally adding gate binding.
 * @param {GPUBuffer} uniformBuffer
 * @param {import('../tensor.js').Tensor} input
 * @param {GPUBuffer} output
 * @param {import('../tensor.js').Tensor | null} gate
 * @returns {GPUBindGroupEntry[]}
 */
function createSiLUBindGroupEntries(uniformBuffer, input, output, gate) {
  /** @type {GPUBindGroupEntry[]} */
  const entries = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: input.buffer } },
    { binding: 2, resource: { buffer: output } },
  ];
  if (gate) {
    entries.push({ binding: 3, resource: { buffer: gate.buffer } });
  }
  return entries;
}

/**
 * Run SiLU activation
 * @param {import('../tensor.js').Tensor} input
 * @param {import('./silu.js').SiLUOptions} [options]
 * @returns {Promise<import('../tensor.js').Tensor>}
 */
export async function runSiLU(
  input,
  options = {}
) {
  const device = getDevice();
  const { size, gate = null, outputBuffer = null, useVec4 = false } = options;

  const isF16 = canUseF16(input);
  const bytesPerElement = dtypeBytes(input.dtype);

  // Select variant using lookup table
  const baseVariant = gate ? 'gate' : (useVec4 ? 'vec4' : 'default');
  const variant = selectSiLUVariant(baseVariant, isF16);
  const pipeline = await getPipelineFast('silu', variant);

  const inferredSize = size || (input.buffer.size / bytesPerElement);
  const outputSize = inferredSize * bytesPerElement;
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

  // Create bind group using helper
  const entries = createSiLUBindGroupEntries(uniformBuffer, input, output, gate);

  const bindGroup = device.createBindGroup({
    label: 'silu_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  const workgroups = Math.ceil(inferredSize / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'silu');

  uniformBuffer.destroy();

  return createTensor(output, input.dtype, [inferredSize], 'silu_output');
}

/**
 * Run SwiGLU with row-split bias
 * @param {import('../tensor.js').Tensor} input
 * @param {import('../tensor.js').Tensor} bias
 * @param {number} numTokens
 * @param {number} dim
 * @param {import('./silu.js').SiLUOptions} [options]
 * @returns {Promise<import('../tensor.js').Tensor>}
 */
export async function runSwiGLURowsplitBias(
  input,
  bias,
  numTokens,
  dim,
  options = {}
) {
  const device = getDevice();
  const { outputBuffer = null, biasOffset = 0 } = options;

  const pipeline = await getPipelineFast('swiglu', 'rowsplit_bias');

  const bytesPerElement = dtypeBytes(input.dtype);
  const outputSize = numTokens * dim * bytesPerElement;
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
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: bias.buffer } },
      { binding: 3, resource: { buffer: output } },
    ],
  });

  const workgroups = Math.ceil((numTokens * dim) / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'swiglu');

  uniformBuffer.destroy();

  return createTensor(output, input.dtype, [numTokens, dim], 'swiglu_output');
}

/**
 * Run row-split SiLU/GELU for fused gate+up FFN.
 *
 * Input: [numTokens, 2*dim] where each row is [gate[0..dim), up[0..dim)]
 * Output: [numTokens, dim] = activation(gate) * up
 * @param {import('../tensor.js').Tensor} input
 * @param {import('./silu.js').SiLURowSplitOptions} options
 * @returns {Promise<import('../tensor.js').Tensor>}
 */
export async function runSiLURowSplit(
  input,
  options
) {
  const device = getDevice();
  const { numTokens, dim, activation = 'silu', outputBuffer = null } = options;

  const isF16 = canUseF16(input);
  const bytesPerElement = dtypeBytes(input.dtype);

  // Select variant (append _f16 for F16 mode)
  let variant = activation === 'gelu' ? 'geglu_rowsplit' : 'gate_rowsplit';
  if (isF16) {
    variant = variant + '_f16';
  }
  const pipeline = await getPipelineFast('silu', variant);

  const outputSize = numTokens * dim * bytesPerElement;
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
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  const workgroups = Math.ceil((numTokens * dim) / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'silu_rowsplit');

  uniformBuffer.destroy();

  return createTensor(output, input.dtype, [numTokens, dim], 'silu_rowsplit_output');
}

/**
 * Record row-split SiLU/GELU (batched, no submit)
 * @param {import('../command-recorder.js').CommandRecorder} recorder
 * @param {import('../tensor.js').Tensor} input
 * @param {import('./silu.js').SiLURowSplitOptions} options
 * @returns {Promise<import('../tensor.js').Tensor>}
 */
export async function recordSiLURowSplit(
  recorder,
  input,
  options
) {
  const device = recorder.device;
  const { numTokens, dim, activation = 'silu', outputBuffer = null } = options;

  const isF16 = canUseF16(input);
  const bytesPerElement = dtypeBytes(input.dtype);

  // Select variant (append _f16 for F16 mode)
  let variant = activation === 'gelu' ? 'geglu_rowsplit' : 'gate_rowsplit';
  if (isF16) {
    variant = variant + '_f16';
  }
  const pipeline = await getPipelineFast('silu', variant);

  const outputSize = numTokens * dim * bytesPerElement;
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
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  const workgroups = Math.ceil((numTokens * dim) / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'silu_rowsplit');

  return createTensor(output, input.dtype, [numTokens, dim], 'silu_rowsplit_output');
}

/**
 * Record SiLU (batched, no submit)
 * Supports gated variant when options.gate is provided.
 * @param {import('../command-recorder.js').CommandRecorder} recorder
 * @param {import('../tensor.js').Tensor} input
 * @param {import('./silu.js').SiLUOptions} [options]
 * @returns {Promise<import('../tensor.js').Tensor>}
 */
export async function recordSiLU(
  recorder,
  input,
  options = {}
) {
  const device = recorder.device;
  const { size, gate = null, outputBuffer = null } = options;

  const isF16 = canUseF16(input);
  const bytesPerElement = dtypeBytes(input.dtype);

  // Select variant using lookup table
  const baseVariant = gate ? 'gate' : 'default';
  const variant = selectSiLUVariant(baseVariant, isF16);
  const pipeline = await getPipelineFast('silu', variant);

  const inferredSize = size || (input.buffer.size / bytesPerElement);
  const outputSize = inferredSize * bytesPerElement;
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

  // Create bind group using helper
  const entries = createSiLUBindGroupEntries(uniformBuffer, input, output, gate);

  const bindGroup = device.createBindGroup({
    label: 'silu_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  const workgroups = Math.ceil(inferredSize / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'silu');

  return createTensor(output, input.dtype, [inferredSize], 'silu_output');
}
