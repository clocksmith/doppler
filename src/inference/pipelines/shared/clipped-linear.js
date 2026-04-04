import { getDevice } from '../../../gpu/device.js';
import { createTensor } from '../../../gpu/tensor.js';
import { runClamp } from '../../../gpu/kernels/clamp.js';
import { runMatmul } from '../../../gpu/kernel-selector.js';
import { acquireBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';

export function shouldClamp(minValue, maxValue) {
  return Number.isFinite(minValue) || Number.isFinite(maxValue);
}

async function cloneTensorBuffer(tensor, label) {
  const device = getDevice();
  const byteLength = tensor.buffer.size;
  const buffer = acquireBuffer(byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, label);
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(tensor.buffer, 0, buffer, 0, byteLength);
    device.queue.submit([encoder.finish()]);
    return createTensor(buffer, tensor.dtype, [...tensor.shape], label);
  } catch (error) {
    releaseBuffer(buffer);
    throw error;
  }
}

export async function runClippableLinear(inputTensor, weight, M, N, K, clip, label) {
  let matmulInput = inputTensor;
  try {
    if (shouldClamp(clip?.inputMin, clip?.inputMax)) {
      matmulInput = await cloneTensorBuffer(inputTensor, `${label}_input`);
      await runClamp(matmulInput, clip.inputMin, clip.inputMax, { count: M * K });
    }
    const output = await runMatmul(matmulInput, weight, M, N, K, {
      outputDtype: 'f32',
      transposeB: 'auto',
    });
    if (shouldClamp(clip?.outputMin, clip?.outputMax)) {
      await runClamp(output, clip.outputMin, clip.outputMax, { count: M * N });
    }
    return output;
  } finally {
    if (matmulInput !== inputTensor) {
      releaseBuffer(matmulInput.buffer);
    }
  }
}
