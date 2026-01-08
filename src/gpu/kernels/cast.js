/**
 * Type Casting Kernels
 *
 * Provides GPU-based type conversions:
 * - F32 to F16
 * - F16 to F32
 * - BF16 to F32
 */

import { getDevice } from '../device.js';
import { acquireBuffer } from '../buffer-pool.js';
import { createTensor } from '../tensor.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView } from './utils.js';
import { GPU_LIMITS, WORKGROUP_SIZES } from './constants.js';
import { trace } from '../../debug/index.js';
import { DTYPE_SIZES } from '../../config/schema/index.js';

// =============================================================================
// Dispatch Helpers
// =============================================================================

/**
 * Calculate 2D dispatch for large workgroup counts.
 * WebGPU has a limit of 65535 workgroups per dimension.
 * @param {number} workgroups
 * @returns {[number, number, number]}
 */
function calculate2DDispatch(workgroups) {
  const maxWorkgroupsPerDim = GPU_LIMITS.MAX_WORKGROUPS;
  return workgroups <= maxWorkgroupsPerDim
    ? [workgroups, 1, 1]
    : [maxWorkgroupsPerDim, Math.ceil(workgroups / maxWorkgroupsPerDim), 1];
}

/**
 * LCM utility for alignment calculations.
 * @param {number} a
 * @param {number} b
 * @returns {number}
 */
function lcm(a, b) {
  /**
   * @param {number} x
   * @param {number} y
   * @returns {number}
   */
  const gcd = (x, y) => {
    let a0 = x;
    let b0 = y;
    while (b0 !== 0) {
      const t = b0;
      b0 = a0 % b0;
      a0 = t;
    }
    return a0;
  };
  return (a / gcd(a, b)) * b;
}

/**
 * Cast F32 buffer to F16 on GPU
 * @param {import('../tensor.js').Tensor} input
 * @param {import('./cast.js').CastOptions} [options]
 * @returns {Promise<import('../tensor.js').Tensor>}
 */
export async function castF32ToF16(
  input,
  options = {}
) {
  const device = getDevice();
  const { outputBuffer = null } = options;
  const numElements = input.shape.reduce((a, b) => a * b, 1);

  const pipeline = await createPipeline('cast', 'f32_to_f16');

  const outputSize = numElements * DTYPE_SIZES.f16;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'cast_f32_to_f16_output');

  const uniformBuffer = createUniformBufferWithView(
    'cast_f32_to_f16_uniforms',
    16,
    (view) => {
      view.setUint32(0, numElements, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'cast_f32_to_f16_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  // Use 2D dispatch for large tensors (like embeddings with 300M+ elements)
  const workgroups = Math.ceil(numElements / WORKGROUP_SIZES.DEFAULT);
  const dispatchSize = calculate2DDispatch(workgroups);

  dispatch(device, pipeline, bindGroup, dispatchSize, 'cast_f32_to_f16');

  // Wait for GPU work to complete before returning
  // Critical for large tensors (embeddings/lm_head) that may take time to convert
  await device.queue.onSubmittedWorkDone();

  uniformBuffer.destroy();

  return createTensor(output, 'f16', [...input.shape], input.label ? `${input.label}_f16` : 'cast_f32_to_f16_output');
}

/**
 * Cast F16 buffer to F32 on GPU
 * @param {import('../tensor.js').Tensor} input
 * @param {import('./cast.js').CastOptions} [options]
 * @returns {Promise<import('../tensor.js').Tensor>}
 */
export async function castF16ToF32(
  input,
  options = {}
) {
  const device = getDevice();
  const { outputBuffer = null } = options;
  const numElements = input.shape.reduce((a, b) => a * b, 1);

  const pipeline = await createPipeline('cast', 'f16_to_f32');

  const outputSize = numElements * DTYPE_SIZES.f32;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'cast_f16_to_f32_output');

  const uniformBuffer = createUniformBufferWithView(
    'cast_f16_to_f32_uniforms',
    16,
    (view) => {
      view.setUint32(0, numElements, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'cast_f16_to_f32_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  const workgroups = Math.ceil(numElements / WORKGROUP_SIZES.DEFAULT);
  const dispatchSize = calculate2DDispatch(workgroups);

  dispatch(device, pipeline, bindGroup, dispatchSize, 'cast_f16_to_f32');

  await device.queue.onSubmittedWorkDone();

  uniformBuffer.destroy();

  return createTensor(output, 'f32', [...input.shape], input.label ? `${input.label}_f32` : 'cast_f16_to_f32_output');
}

/**
 * Record F32 to F16 cast (batched, no submit)
 * @param {import('../command-recorder.js').CommandRecorder} recorder
 * @param {import('../tensor.js').Tensor} input
 * @param {import('./cast.js').CastOptions} [options]
 * @returns {Promise<import('../tensor.js').Tensor>}
 */
export async function recordCastF32ToF16(
  recorder,
  input,
  options = {}
) {
  const device = recorder.device;
  const { outputBuffer = null } = options;
  const numElements = input.shape.reduce((a, b) => a * b, 1);

  const pipeline = await createPipeline('cast', 'f32_to_f16');

  const outputSize = numElements * DTYPE_SIZES.f16;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'cast_f32_to_f16_output');

  const uniformBuffer = createUniformBufferWithView(
    'cast_f32_to_f16_uniforms',
    16,
    (view) => {
      view.setUint32(0, numElements, true);
    },
    recorder
  );

  const bindGroup = device.createBindGroup({
    label: 'cast_f32_to_f16_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  // Use 2D dispatch for large tensors
  const workgroups = Math.ceil(numElements / WORKGROUP_SIZES.DEFAULT);
  const dispatchSize = calculate2DDispatch(workgroups);

  recordDispatch(recorder, pipeline, bindGroup, dispatchSize, 'cast_f32_to_f16');

  return createTensor(output, 'f16', [...input.shape], input.label ? `${input.label}_f16` : 'cast_f32_to_f16_output');
}

/**
 * Record F16 to F32 cast (batched, no submit)
 * @param {import('../command-recorder.js').CommandRecorder} recorder
 * @param {import('../tensor.js').Tensor} input
 * @param {import('./cast.js').CastOptions} [options]
 * @returns {Promise<import('../tensor.js').Tensor>}
 */
export async function recordCastF16ToF32(
  recorder,
  input,
  options = {}
) {
  const device = recorder.device;
  const { outputBuffer = null } = options;
  const numElements = input.shape.reduce((a, b) => a * b, 1);

  const pipeline = await createPipeline('cast', 'f16_to_f32');

  const outputSize = numElements * DTYPE_SIZES.f32;
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'cast_f16_to_f32_output');

  const uniformBuffer = createUniformBufferWithView(
    'cast_f16_to_f32_uniforms',
    16,
    (view) => {
      view.setUint32(0, numElements, true);
    },
    recorder
  );

  const bindGroup = device.createBindGroup({
    label: 'cast_f16_to_f32_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  const workgroups = Math.ceil(numElements / WORKGROUP_SIZES.DEFAULT);
  const dispatchSize = calculate2DDispatch(workgroups);

  recordDispatch(recorder, pipeline, bindGroup, dispatchSize, 'cast_f16_to_f32');

  return createTensor(output, 'f32', [...input.shape], input.label ? `${input.label}_f32` : 'cast_f16_to_f32_output');
}

/**
 * Convert BF16 buffer to F32 on GPU
 * @param {GPUBuffer} input
 * @param {readonly number[]} shape
 * @param {string} [name]
 * @returns {Promise<import('../tensor.js').Tensor>}
 */
export async function runBF16ToF32(
  input,
  shape,
  name = 'bf16_to_f32_output'
) {
  const numElements = shape.reduce((a, b) => a * b, 1);
  trace.kernels(`BF16ToF32: Entry numElements=${numElements}, name=${name}, inputSize=${input.size}`);
  const device = getDevice();

  // Check for size limits (handle chunking if needed)
  const limits = device.limits;
  const maxBufferSize = limits.maxBufferSize;
  const maxBindingSize = limits.maxStorageBufferBindingSize;
  const outputSize = numElements * DTYPE_SIZES.f32;
  trace.kernels(`BF16ToF32: outputSize=${outputSize}, maxBufferSize=${maxBufferSize}, maxBindingSize=${maxBindingSize}`);

  if (outputSize > maxBufferSize) {
    throw new Error(
      `BF16→F32 output (${outputSize} bytes) exceeds device maxBufferSize (${maxBufferSize}). ` +
      `This often happens for large-vocab models when converting embeddings/LM head. ` +
      `Enable F16 and use BF16→F16 weights, or run on a device with a higher maxBufferSize.`
    );
  }

  if (outputSize > maxBindingSize) {
    // Need to chunk - output buffer can exist, but must be bound in smaller ranges.
    return runBF16ToF32Chunked(input, shape, name, maxBindingSize);
  }

  const pipeline = await createPipeline('bf16_to_f32', 'default');
  trace.kernels('BF16ToF32: Pipeline created');

  const output = acquireBuffer(outputSize, undefined, name);
  trace.kernels(`BF16ToF32: Output buffer acquired, size=${output.size}`);

  const uniformBuffer = createUniformBufferWithView(
    'bf16_to_f32_uniforms',
    16,
    (view) => {
      view.setUint32(0, numElements, true);
    },
    null,
    device
  );
  trace.kernels(`BF16ToF32: Uniform numElements=${numElements}`);

  const bindGroup = device.createBindGroup({
    label: 'bf16_to_f32_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } },
    ],
  });
  trace.kernels('BF16ToF32: BindGroup created');

  // Each thread processes 2 BF16 values (1 u32), so divide by 2 for thread count
  // Then divide by 256 for workgroup count
  const numPairs = Math.ceil(numElements / 2);
  const workgroups = Math.ceil(numPairs / WORKGROUP_SIZES.DEFAULT);
  const dispatchSize = calculate2DDispatch(workgroups);

  trace.kernels(`BF16ToF32: Dispatching ${dispatchSize[0]}x${dispatchSize[1]} workgroups for ${numPairs} pairs (${numElements} elements)`);
  dispatch(device, pipeline, bindGroup, dispatchSize, 'bf16_to_f32');

  // Wait for GPU work to complete before returning
  await device.queue.onSubmittedWorkDone();
  trace.kernels('BF16ToF32: GPU work completed');

  uniformBuffer.destroy();

  return createTensor(output, 'f32', [...shape], name);
}

/**
 * Convert BF16 buffer to F16 on GPU (no intermediate F32 buffer)
 * @param {GPUBuffer} input
 * @param {readonly number[]} shape
 * @param {string} [name]
 * @returns {Promise<import('../tensor.js').Tensor>}
 */
export async function runBF16ToF16(
  input,
  shape,
  name = 'bf16_to_f16_output'
) {
  const numElements = shape.reduce((a, b) => a * b, 1);
  const device = getDevice();
  const pipeline = await createPipeline('bf16_to_f16', 'default');

  const limits = device.limits;
  const maxBufferSize = limits.maxBufferSize;
  const maxBindingSize = limits.maxStorageBufferBindingSize;
  const outputSize = numElements * DTYPE_SIZES.f16;

  if (outputSize > maxBufferSize) {
    throw new Error(
      `BF16→F16 output (${outputSize} bytes) exceeds device maxBufferSize (${maxBufferSize}).`
    );
  }
  if (outputSize > maxBindingSize) {
    throw new Error(
      `BF16→F16 output (${outputSize} bytes) exceeds device maxStorageBufferBindingSize (${maxBindingSize}).`
    );
  }

  const output = acquireBuffer(outputSize, undefined, name);

  const uniformBuffer = createUniformBufferWithView(
    'bf16_to_f16_uniforms',
    16,
    (view) => {
      view.setUint32(0, numElements, true);
      view.setUint32(4, 0, true);
      view.setUint32(8, 0, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'bf16_to_f16_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  const numPairs = Math.ceil(numElements / 2);
  const workgroups = Math.ceil(numPairs / WORKGROUP_SIZES.DEFAULT);
  const dispatchSize = calculate2DDispatch(workgroups);

  dispatch(device, pipeline, bindGroup, dispatchSize, 'bf16_to_f16');
  await device.queue.onSubmittedWorkDone();

  uniformBuffer.destroy();

  return createTensor(output, 'f16', [...shape], name);
}

/**
 * Convert BF16 to F32 in chunks (for large buffers)
 * @param {GPUBuffer} input
 * @param {readonly number[]} shape
 * @param {string} name
 * @param {number} maxBindingSize
 * @returns {Promise<import('../tensor.js').Tensor>}
 */
async function runBF16ToF32Chunked(
  input,
  shape,
  name,
  maxBindingSize
) {
  const numElements = shape.reduce((a, b) => a * b, 1);
  const device = getDevice();
  const pipeline = await createPipeline('bf16_to_f32', 'default');

  // Calculate chunk size
  const alignmentBytes = device.limits.minStorageBufferOffsetAlignment;

  const inElemAlign = Math.max(1, Math.floor(alignmentBytes / DTYPE_SIZES.bf16)); // BF16 elements
  const outElemAlign = Math.max(1, Math.floor(alignmentBytes / DTYPE_SIZES.f32)); // F32 elements
  const elemAlign = lcm(inElemAlign, outElemAlign);

  let maxElementsPerChunk = Math.floor(maxBindingSize / DTYPE_SIZES.f32); // F32 output bytes
  maxElementsPerChunk -= maxElementsPerChunk % elemAlign;
  if (maxElementsPerChunk <= 0) {
    throw new Error(`BF16→F32 chunk size underflow (maxBindingSize=${maxBindingSize}, alignment=${alignmentBytes})`);
  }
  const numChunks = Math.ceil(numElements / maxElementsPerChunk);

  // Create full output buffer
  const outputSize = numElements * DTYPE_SIZES.f32;
  const output = acquireBuffer(outputSize, undefined, name);

  trace.kernels(`BF16ToF32: Chunking ${numElements} elements in ${numChunks} chunks`);

  for (let chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
    const chunkStart = chunkIdx * maxElementsPerChunk;
    const chunkEnd = Math.min((chunkIdx + 1) * maxElementsPerChunk, numElements);
    const chunkSize = chunkEnd - chunkStart;

    const uniformBuffer = createUniformBufferWithView(
      `bf16_to_f32_chunk${chunkIdx}_uniforms`,
      16,
      (view) => {
        view.setUint32(0, chunkSize, true);
        view.setUint32(4, 0, true);
        view.setUint32(8, 0, true);
      },
      null,
      device
    );

    const inputOffsetBytes = chunkStart * DTYPE_SIZES.bf16;
    const outputOffsetBytes = chunkStart * DTYPE_SIZES.f32;
    const inputPairs = Math.ceil(chunkSize / 2);
    const inputSizeBytes = inputPairs * DTYPE_SIZES.f32; // Pairs read as u32
    const outputSizeBytes = chunkSize * DTYPE_SIZES.f32;

    const bindGroup = device.createBindGroup({
      label: `bf16_to_f32_chunk${chunkIdx}_bind_group`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: input, offset: inputOffsetBytes, size: inputSizeBytes } },
        { binding: 2, resource: { buffer: output, offset: outputOffsetBytes, size: outputSizeBytes } },
      ],
    });

    // Each thread processes 2 BF16 values
    const numPairs = Math.ceil(chunkSize / 2);
    const workgroups = Math.ceil(numPairs / WORKGROUP_SIZES.DEFAULT);
    const dispatchSize = calculate2DDispatch(workgroups);

    dispatch(device, pipeline, bindGroup, dispatchSize, `bf16_to_f32_chunk${chunkIdx}`);

    uniformBuffer.destroy();
  }

  return createTensor(output, 'f32', [...shape], name);
}
