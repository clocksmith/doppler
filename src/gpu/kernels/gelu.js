/**
 * GeLU Activation Kernels
 *
 * Provides GeLU activation: x * Phi(x) where Phi is the CDF of standard normal distribution.
 */

import { getDevice } from '../device.js';
import { acquireBuffer } from '../buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView } from './utils.js';

/**
 * Check if F16 can be used based on tensor dtype.
 * @param {Tensor} input
 * @returns {boolean}
 */
function canUseF16(input) {
  return input.dtype === 'f16';
}

/**
 * Run GeLU activation
 * @param {Tensor} input
 * @param {import('./gelu.js').GeLUOptions} [options]
 * @returns {Promise<Tensor>}
 */
export async function runGeLU(
  input,
  options = {}
) {
  const device = getDevice();
  const { size, gate = null, outputBuffer = null } = options;

  const isF16 = canUseF16(input);
  const bytesPerElement = dtypeBytes(input.dtype);

  // Select gated variant when gate buffer is provided
  // Use F16 variants when dtype is 'f16' (geglu_rowsplit_f16 in silu_f16.wgsl)
  /** @type {string} */
  let variant;
  if (gate) {
    variant = isF16 ? 'geglu_rowsplit_f16' : 'geglu';
  } else {
    variant = 'gelu';  // No F16 gelu (non-gated) variant - use F32
  }
  const pipeline = await createPipeline('silu', variant);

  const inferredSize = size || (input.buffer.size / bytesPerElement);
  const outputSize = inferredSize * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'gelu_output');

  // Create uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'gelu_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredSize, true);
    },
    null,
    device
  );

  // Create bind group
  // WGSL bindings: 0=uniforms, 1=input, 2=output, 3=gate, 4=bias
  const gateBuffer = gate ? gate.buffer : input.buffer;
  const bindGroup = device.createBindGroup({
    label: 'gelu_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: gateBuffer } },
    ],
  });

  const workgroups = Math.ceil(inferredSize / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'gelu');

  uniformBuffer.destroy();

  return createTensor(output, input.dtype, [inferredSize], 'gelu_output');
}

/**
 * Record GeLU (batched, no submit)
 * Supports gated variant (GeGLU) when options.gate is provided.
 * @param {import('../command-recorder.js').CommandRecorder} recorder
 * @param {Tensor} input
 * @param {import('./gelu.js').GeLUOptions} [options]
 * @returns {Promise<Tensor>}
 */
export async function recordGeLU(
  recorder,
  input,
  options = {}
) {
  const device = recorder.device;
  const { size, gate = null, outputBuffer = null } = options;

  const isF16 = canUseF16(input);
  const bytesPerElement = dtypeBytes(input.dtype);

  // Select gated variant when gate buffer is provided
  // Use F16 variants when dtype is 'f16' (geglu_rowsplit_f16 in silu_f16.wgsl)
  /** @type {string} */
  let variant;
  if (gate) {
    variant = isF16 ? 'geglu_rowsplit_f16' : 'geglu';
  } else {
    variant = 'gelu';  // No F16 gelu (non-gated) variant - use F32
  }
  const pipeline = await createPipeline('silu', variant);

  const inferredSize = size || (input.buffer.size / bytesPerElement);
  const outputSize = inferredSize * bytesPerElement;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'gelu_output');

  // Uniform buffer
  const uniformBuffer = createUniformBufferWithView(
    'gelu_uniforms',
    16,
    (view) => {
      view.setUint32(0, inferredSize, true);
    },
    recorder
  );

  // Bind group entries - gate variant needs binding 3
  /** @type {GPUBindGroupEntry[]} */
  const entries = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: input.buffer } },
    { binding: 2, resource: { buffer: output } },
  ];

  // Add gate binding for gated variant
  if (gate) {
    entries.push({ binding: 3, resource: { buffer: gate.buffer } });
  }

  const bindGroup = device.createBindGroup({
    label: 'gelu_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  const workgroups = Math.ceil(inferredSize / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'gelu');

  return createTensor(output, input.dtype, [inferredSize], 'gelu_output');
}
