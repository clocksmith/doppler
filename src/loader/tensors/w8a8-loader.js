import { getDevice } from '../../gpu/device.js';
import { releaseBuffer } from '../../memory/buffer-pool.js';
import { createWeightBuffer } from '../../gpu/weight-buffer.js';
import { validateW8A8Policy } from '../../config/schema/w8a8.schema.js';
import { validateW8A8StorageBytes } from '../../formats/rdrr/w8a8-storage.js';
import { acquireAlignedBuffer, writeBufferAligned } from './load-executor.js';

/**
 * @param {Uint8Array} shardData
 * @param {import('../loader-types.js').TensorLocation} location
 * @param {string} name
 * @param {import('./tensor-loader.js').TensorLoadConfig} config
 * @returns {Promise<import('./tensor-loader.js').TensorLoadResult>}
 */
export async function loadW8A8Packed(shardData, location, name, config) {
  validateW8A8Policy(config?.w8a8Policy, 'manifest.quantizationInfo.w8a8');
  if (location.dtype !== 'W8A8' || location.role !== 'matmul' || location.layout !== 'row') {
    throw new Error(`W8A8 tensor ${name} must declare row-major matmul storage.`);
  }
  const storage = validateW8A8StorageBytes(shardData, location.shape, location.storage);
  const buffer = acquireAlignedBuffer(shardData.byteLength, `w8a8_${name}`);
  try {
    writeBufferAligned(getDevice(), buffer, shardData);
    return {
      data: createWeightBuffer(buffer, 'w8a8', 'row', location.shape, name, null, {
        packing: storage.packing, paddedK: storage.paddedK,
        rowStrideBytes: storage.rowStrideBytes,
        scaleOffsetBytes: storage.scaleOffsetBytes, dataOffsetBytes: storage.dataOffsetBytes,
      }),
      allocatedBuffers: [buffer],
    };
  } catch (error) {
    releaseBuffer(buffer);
    throw error;
  }
}
