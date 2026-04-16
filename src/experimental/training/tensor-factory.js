import { acquireBuffer, releaseBuffer, uploadData } from '../../memory/buffer-pool.js';
import { createTensor } from '../../gpu/tensor.js';

export function createUploadedTensor(data, dtype, shape, label, usage = undefined) {
  const buffer = acquireBuffer(data.byteLength, usage, label);
  try {
    uploadData(buffer, data);
    return createTensor(buffer, dtype, [...shape], label);
  } catch (error) {
    releaseBuffer(buffer);
    throw error;
  }
}
