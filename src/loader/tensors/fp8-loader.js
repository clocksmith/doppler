import { getDevice } from '../../gpu/device.js';
import { createWeightBuffer } from '../../gpu/weight-buffer.js';
import { releaseBuffer } from '../../memory/buffer-pool.js';
import { validateFP8Policy } from '../../config/schema/fp8.schema.js';
import { validateFP8GPUStorageBytes } from '../../formats/rdrr/fp8-storage.js';
import { acquireAlignedBuffer, writeBufferAligned } from './load-executor.js';

/** @type {typeof import('./fp8-loader.js').loadFP8Packed} */
export async function loadFP8Packed(bytes, location, name, config) {
  const policy = validateFP8Policy(config.fp8Policy, 'manifest.quantizationInfo.fp8');
  const storage = validateFP8GPUStorageBytes(bytes, location.shape, location.storage);
  const expected = policy.encoding === 'e4m3fn' ? 'FP8_E4M3FN' : 'FP8_E5M2';
  if (location.dtype !== expected || location.role !== 'matmul' || location.layout !== 'row'
    || storage.encoding !== policy.encoding || JSON.stringify(storage.blockShape) !== JSON.stringify(policy.blockShape)) {
    throw new Error(`FP8 tensor ${name} differs from its declared packed matmul policy.`);
  }
  if (!config.representationRequirement?.complete || config.representationRequirement.mode !== 'packed-only'
    || config.representationRequirement.dtypes.length !== 1
    || config.representationRequirement.dtypes[0] !== (policy.encoding === 'e4m3fn' ? 'fp8e4' : 'fp8e5')) {
    throw new Error('FP8 loading requires every declared consumer to accept the same packed encoding.');
  }
  const buffer = acquireAlignedBuffer(bytes.byteLength, `fp8_${name}`);
  try {
    writeBufferAligned(getDevice(), buffer, bytes);
    return { data: createWeightBuffer(buffer, policy.encoding === 'e4m3fn' ? 'fp8e4' : 'fp8e5', 'row', location.shape, name, null, storage),
      allocatedBuffers: [buffer] };
  } catch (error) { releaseBuffer(buffer); throw error; }
}
