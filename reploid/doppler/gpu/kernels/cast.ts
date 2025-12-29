/**
 * Type Casting Kernels
 *
 * Provides GPU-based type conversions:
 * - F32 to F16
 * - BF16 to F32
 */

import { getDevice } from '../device.js';
import { setBufferDtype } from '../buffer-dtypes.js';
import { acquireBuffer } from '../buffer-pool.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView } from './utils.js';
import { allowReadback } from '../perf-guards.js';
import type { CommandRecorder } from '../command-recorder.js';
import { GPU_LIMITS, WORKGROUP_SIZES } from './constants.js';
import type { OutputBufferOptions } from './types.js';

/** Debug flag for cast kernel logging */
const DEBUG_CAST = false;

/** Cast kernel options */
export interface CastOptions extends OutputBufferOptions {}

/**
 * Cast F32 buffer to F16 on GPU
 */
export async function castF32ToF16(
  input: GPUBuffer,
  numElements: number,
  options: CastOptions = {}
): Promise<GPUBuffer> {
  const device = getDevice();
  const { outputBuffer = null } = options;

  const pipeline = await createPipeline('cast', 'f32_to_f16');

  const output = outputBuffer || acquireBuffer(numElements * 2, undefined, 'cast_f32_to_f16_output');

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
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  // WebGPU has a limit of 65535 workgroups per dimension
  // Use 2D dispatch for large tensors (like embeddings with 300M+ elements)
  const workgroups = Math.ceil(numElements / WORKGROUP_SIZES.DEFAULT);
  const maxWorkgroupsPerDim = GPU_LIMITS.MAX_WORKGROUPS;
  const dispatchSize: [number, number, number] = workgroups <= maxWorkgroupsPerDim
    ? [workgroups, 1, 1]
    : [maxWorkgroupsPerDim, Math.ceil(workgroups / maxWorkgroupsPerDim), 1];

  dispatch(device, pipeline, bindGroup, dispatchSize, 'cast_f32_to_f16');

  // Wait for GPU work to complete before returning
  // Critical for large tensors (embeddings/lm_head) that may take time to convert
  await device.queue.onSubmittedWorkDone();

  uniformBuffer.destroy();

  setBufferDtype(output, 'f16');
  return output;
}

/**
 * Record F32 to F16 cast (batched, no submit)
 */
export async function recordCastF32ToF16(
  recorder: CommandRecorder,
  input: GPUBuffer,
  numElements: number,
  options: CastOptions = {}
): Promise<GPUBuffer> {
  const device = recorder.device;
  const { outputBuffer = null } = options;

  const pipeline = await createPipeline('cast', 'f32_to_f16');

  const output = outputBuffer || acquireBuffer(numElements * 2, undefined, 'cast_f32_to_f16_output');

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
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } },
    ],
  });

  // WebGPU has a limit of 65535 workgroups per dimension
  // Use 2D dispatch for large tensors
  const workgroups = Math.ceil(numElements / WORKGROUP_SIZES.DEFAULT);
  const maxWorkgroupsPerDim = GPU_LIMITS.MAX_WORKGROUPS;
  const dispatchSize: [number, number, number] = workgroups <= maxWorkgroupsPerDim
    ? [workgroups, 1, 1]
    : [maxWorkgroupsPerDim, Math.ceil(workgroups / maxWorkgroupsPerDim), 1];

  recordDispatch(recorder, pipeline, bindGroup, dispatchSize, 'cast_f32_to_f16');

  setBufferDtype(output, 'f16');
  return output;
}

/**
 * Convert BF16 buffer to F32 on GPU
 */
export async function runBF16ToF32(
  input: GPUBuffer,
  numElements: number,
  name: string = 'bf16_to_f32_output'
): Promise<GPUBuffer> {
  if (DEBUG_CAST) console.log(`[BF16ToF32] Entry: numElements=${numElements}, name=${name}, inputSize=${input.size}`);
  const device = getDevice();

  // Check for size limits (handle chunking if needed)
  const limits = device.limits;
  const maxBufferSize = limits.maxBufferSize;
  const maxBindingSize = limits.maxStorageBufferBindingSize;
  const outputSize = numElements * 4; // F32
  if (DEBUG_CAST) console.log(`[BF16ToF32] outputSize=${outputSize}, maxBufferSize=${maxBufferSize}, maxBindingSize=${maxBindingSize}`);

  if (outputSize > maxBufferSize) {
    throw new Error(
      `BF16→F32 output (${outputSize} bytes) exceeds device maxBufferSize (${maxBufferSize}). ` +
      `This often happens for large-vocab models when converting embeddings/LM head. ` +
      `Enable F16 and use BF16→F16 weights, or run on a device with a higher maxBufferSize.`
    );
  }

  if (outputSize > maxBindingSize) {
    // Need to chunk - output buffer can exist, but must be bound in smaller ranges.
    return runBF16ToF32Chunked(input, numElements, name, maxBindingSize);
  }

  // DEBUG: Verify input buffer has non-zero data before conversion
  if (name.includes('embed') && allowReadback('cast.debugInput')) {
    try {
      const sampleSize = Math.min(256, input.size);
      const stagingIn = device.createBuffer({
        size: sampleSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        label: 'bf16_input_debug_staging',
      });
      const encIn = device.createCommandEncoder();
      encIn.copyBufferToBuffer(input, 0, stagingIn, 0, sampleSize);
      device.queue.submit([encIn.finish()]);
      await device.queue.onSubmittedWorkDone();
      await stagingIn.mapAsync(GPUMapMode.READ);
      const inData = new Uint8Array(stagingIn.getMappedRange().slice(0));
      stagingIn.unmap();
      stagingIn.destroy();
      const nonZeroBytes = Array.from(inData).filter(x => x !== 0).length;
      console.log(`[BF16ToF32] INPUT CHECK: nonZeroBytes=${nonZeroBytes}/${inData.length}, first16=[${Array.from(inData.slice(0, 16)).join(', ')}]`);
    } catch (err) {
      console.error(`[BF16ToF32] INPUT CHECK failed:`, (err as Error).message);
    }
  }

  const pipeline = await createPipeline('bf16_to_f32', 'default');
  if (DEBUG_CAST) console.log(`[BF16ToF32] Pipeline created`);

  const output = acquireBuffer(outputSize, undefined, name);
  if (DEBUG_CAST) console.log(`[BF16ToF32] Output buffer acquired, size=${output.size}`);

  const uniformBuffer = createUniformBufferWithView(
    'bf16_to_f32_uniforms',
    16,
    (view) => {
      view.setUint32(0, numElements, true);
    },
    null,
    device
  );
  if (DEBUG_CAST) console.log(`[BF16ToF32] Uniform: numElements=${numElements}`);

  const bindGroup = device.createBindGroup({
    label: 'bf16_to_f32_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: input } },
      { binding: 2, resource: { buffer: output } },
    ],
  });
  if (DEBUG_CAST) console.log(`[BF16ToF32] BindGroup created`);

  // Each thread processes 2 BF16 values (1 u32), so divide by 2 for thread count
  // Then divide by 256 for workgroup count
  const numPairs = Math.ceil(numElements / 2);
  const workgroups = Math.ceil(numPairs / WORKGROUP_SIZES.DEFAULT);

  // WebGPU has a limit of 65535 workgroups per dimension
  // Use 2D dispatch for large tensors
  const maxWorkgroupsPerDim = GPU_LIMITS.MAX_WORKGROUPS;
  const dispatchSize: [number, number, number] = workgroups <= maxWorkgroupsPerDim
    ? [workgroups, 1, 1]
    : [maxWorkgroupsPerDim, Math.ceil(workgroups / maxWorkgroupsPerDim), 1];

  if (DEBUG_CAST) console.log(
    `[BF16ToF32] Dispatching ${dispatchSize[0]}x${dispatchSize[1]} workgroups ` +
    `for ${numPairs} pairs (${numElements} elements)`
  );
  dispatch(device, pipeline, bindGroup, dispatchSize, 'bf16_to_f32');

  // Wait for GPU work to complete before returning
  await device.queue.onSubmittedWorkDone();
  if (DEBUG_CAST) console.log(`[BF16ToF32] GPU work completed`);

  uniformBuffer.destroy();

  setBufferDtype(output, 'f32');
  return output;
}

/**
 * Convert BF16 buffer to F16 on GPU (no intermediate F32 buffer)
 */
export async function runBF16ToF16(
  input: GPUBuffer,
  numElements: number,
  name: string = 'bf16_to_f16_output'
): Promise<GPUBuffer> {
  const device = getDevice();
  const pipeline = await createPipeline('bf16_to_f16', 'default');

  const limits = device.limits;
  const maxBufferSize = limits.maxBufferSize;
  const maxBindingSize = limits.maxStorageBufferBindingSize;
  const outputSize = numElements * 2; // F16

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
  const maxWorkgroupsPerDim = GPU_LIMITS.MAX_WORKGROUPS;
  const dispatchSize: [number, number, number] = workgroups <= maxWorkgroupsPerDim
    ? [workgroups, 1, 1]
    : [maxWorkgroupsPerDim, Math.ceil(workgroups / maxWorkgroupsPerDim), 1];

  dispatch(device, pipeline, bindGroup, dispatchSize, 'bf16_to_f16');
  await device.queue.onSubmittedWorkDone();

  uniformBuffer.destroy();

  setBufferDtype(output, 'f16');
  return output;
}

/**
 * Convert BF16 to F32 in chunks (for large buffers)
 */
async function runBF16ToF32Chunked(
  input: GPUBuffer,
  numElements: number,
  name: string,
  maxBindingSize: number
): Promise<GPUBuffer> {
  const device = getDevice();
  const pipeline = await createPipeline('bf16_to_f32', 'default');

  // Calculate chunk size
  const alignmentBytes = device.limits.minStorageBufferOffsetAlignment;
  const lcm = (a: number, b: number): number => {
    const gcd = (x: number, y: number): number => {
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
  };

  const inElemAlign = Math.max(1, Math.floor(alignmentBytes / 2)); // BF16 elements
  const outElemAlign = Math.max(1, Math.floor(alignmentBytes / 4)); // F32 elements
  const elemAlign = lcm(inElemAlign, outElemAlign);

  let maxElementsPerChunk = Math.floor(maxBindingSize / 4); // F32 output bytes
  maxElementsPerChunk -= maxElementsPerChunk % elemAlign;
  if (maxElementsPerChunk <= 0) {
    throw new Error(`BF16→F32 chunk size underflow (maxBindingSize=${maxBindingSize}, alignment=${alignmentBytes})`);
  }
  const numChunks = Math.ceil(numElements / maxElementsPerChunk);

  // Create full output buffer
  const outputSize = numElements * 4;
  const output = acquireBuffer(outputSize, undefined, name);

  if (DEBUG_CAST) console.log(`[BF16ToF32] Chunking: ${numElements} elements in ${numChunks} chunks`);

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

    const inputOffsetBytes = chunkStart * 2;
    const outputOffsetBytes = chunkStart * 4;
    const inputPairs = Math.ceil(chunkSize / 2);
    const inputSizeBytes = inputPairs * 4;
    const outputSizeBytes = chunkSize * 4;

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

    // Use 2D dispatch for large chunks
    const maxWorkgroupsPerDim = GPU_LIMITS.MAX_WORKGROUPS;
    const dispatchSize: [number, number, number] = workgroups <= maxWorkgroupsPerDim
      ? [workgroups, 1, 1]
      : [maxWorkgroupsPerDim, Math.ceil(workgroups / maxWorkgroupsPerDim), 1];

    dispatch(device, pipeline, bindGroup, dispatchSize, `bf16_to_f32_chunk${chunkIdx}`);

    uniformBuffer.destroy();
  }

  setBufferDtype(output, 'f32');
  return output;
}
