

import { getDevice } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView } from './utils.js';
import { GPU_LIMITS, WORKGROUP_SIZES } from './constants.js';
import { trace } from '../../debug/index.js';
import { DTYPE_SIZES } from '../../config/schema/index.js';

// =============================================================================
// Dispatch Helpers
// =============================================================================


function calculate2DDispatch(workgroups) {
  const maxWorkgroupsPerDim = GPU_LIMITS.MAX_WORKGROUPS;
  return workgroups <= maxWorkgroupsPerDim
    ? [workgroups, 1, 1]
    : [maxWorkgroupsPerDim, Math.ceil(workgroups / maxWorkgroupsPerDim), 1];
}


function lcm(a, b) {
  
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


async function executeCast(recorder, input, fromDtype, toDtype, options = {}) {
  const device = recorder?.device || getDevice();
  const { outputBuffer = null } = options;
  const ownsOutput = outputBuffer == null;
  const numElements = input.shape.reduce((a, b) => a * b, 1);

  const variant = `${fromDtype}_to_${toDtype}`;
  const pipeline = await createPipeline('cast', variant);

  const outputSize = numElements * DTYPE_SIZES[toDtype];
  const output = outputBuffer || acquireBuffer(outputSize, undefined, `cast_${variant}_output`);

  let uniformBuffer = null;
  try {
    uniformBuffer = createUniformBufferWithView(
      `cast_${variant}_uniforms`,
      16,
      (view) => {
        view.setUint32(0, numElements, true);
      },
      recorder,
      recorder ? undefined : device
    );

    const bindGroup = device.createBindGroup({
      label: `cast_${variant}_bind_group`,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: input.buffer } },
        { binding: 2, resource: { buffer: output } },
      ],
    });

    const workgroups = Math.ceil(numElements / WORKGROUP_SIZES.DEFAULT);
    const dispatchSize = calculate2DDispatch(workgroups);

    if (recorder) {
      recordDispatch(recorder, pipeline, bindGroup, dispatchSize, `cast_${variant}`);
    } else {
      dispatch(device, pipeline, bindGroup, dispatchSize, `cast_${variant}`);
    }
    const label = input.label ? `${input.label}_${toDtype}` : `cast_${variant}_output`;
    return createTensor(output, toDtype, [...input.shape], label);
  } catch (error) {
    if (ownsOutput) {
      releaseBuffer(output);
    }
    throw error;
  } finally {
    if (!recorder) {
      uniformBuffer?.destroy();
    }
  }
}


export async function castF32ToF16(input, options = {}) {
  return executeCast(null, input, 'f32', 'f16', options);
}


export async function castF16ToF32(input, options = {}) {
  return executeCast(null, input, 'f16', 'f32', options);
}


export async function recordCastF32ToF16(recorder, input, options = {}) {
  return executeCast(recorder, input, 'f32', 'f16', options);
}


export async function recordCastF16ToF32(recorder, input, options = {}) {
  return executeCast(recorder, input, 'f16', 'f32', options);
}


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
      `BF16->F32 output (${outputSize} bytes) exceeds device maxBufferSize (${maxBufferSize}). ` +
      `This often happens for large-vocab models when converting embeddings/LM head. ` +
      `Enable F16 and use BF16->F16 weights, or run on a device with a higher maxBufferSize.`
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
  try {
    dispatch(device, pipeline, bindGroup, dispatchSize, 'bf16_to_f32');
    return createTensor(output, 'f32', [...shape], name);
  } catch (error) {
    releaseBuffer(output);
    throw error;
  } finally {
    uniformBuffer.destroy();
  }
}


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
      `BF16->F16 output (${outputSize} bytes) exceeds device maxBufferSize (${maxBufferSize}).`
    );
  }
  if (outputSize > maxBindingSize) {
    throw new Error(
      `BF16->F16 output (${outputSize} bytes) exceeds device maxStorageBufferBindingSize (${maxBindingSize}).`
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

  try {
    dispatch(device, pipeline, bindGroup, dispatchSize, 'bf16_to_f16');
    return createTensor(output, 'f16', [...shape], name);
  } catch (error) {
    releaseBuffer(output);
    throw error;
  } finally {
    uniformBuffer.destroy();
  }
}


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
    throw new Error(`BF16->F32 chunk size underflow (maxBindingSize=${maxBindingSize}, alignment=${alignmentBytes})`);
  }
  const numChunks = Math.ceil(numElements / maxElementsPerChunk);

  // Create full output buffer
  const outputSize = numElements * DTYPE_SIZES.f32;
  const output = acquireBuffer(outputSize, undefined, name);

  trace.kernels(`BF16ToF32: Chunking ${numElements} elements in ${numChunks} chunks`);

  try {
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

      try {
        const inputOffsetBytes = chunkStart * DTYPE_SIZES.bf16;
        const outputOffsetBytes = chunkStart * DTYPE_SIZES.f32;
        const inputPairs = Math.ceil(chunkSize / 2);
        const inputSizeBytes = inputPairs * DTYPE_SIZES.f32;
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

        const numPairs = Math.ceil(chunkSize / 2);
        const workgroups = Math.ceil(numPairs / WORKGROUP_SIZES.DEFAULT);
        const dispatchSize = calculate2DDispatch(workgroups);

        dispatch(device, pipeline, bindGroup, dispatchSize, `bf16_to_f32_chunk${chunkIdx}`);
      } finally {
        uniformBuffer.destroy();
      }
    }

    return createTensor(output, 'f32', [...shape], name);
  } catch (error) {
    releaseBuffer(output);
    throw error;
  }
}
