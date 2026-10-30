import { getDevice } from '../../gpu/device.js';
import { createWeightBuffer } from '../../gpu/weight-buffer.js';
import { releaseBuffer } from '../../memory/buffer-pool.js';
import { ggufBlockFormat,ggufBlockBytes } from '../../formats/gguf/block-storage.js';
import { acquireAlignedBuffer, writeBufferAligned, toUint8View } from './load-executor.js';

/** @type {typeof import('./q4_0-loader.js').loadQ40Packed} */
export function loadQ40Packed(payload, location, name, config) {
  const bytes = toUint8View(payload);
  const format=ggufBlockFormat(location.dtype);
  const size=ggufBlockBytes(location.shape,location.dtype);
  if (location.layout != null && location.layout !== 'row') throw new Error('Packed GGUF requires row-major storage.');
  if (location.storage && (location.storage.blockBytes !== format.bytes || location.storage.blockShape?.[0] !== format.values)) throw new Error('Packed GGUF block descriptor disagrees with dtype.');
  if (bytes.byteLength !== size || (location.size != null && location.size !== size)) {
    throw new Error(`Q4_0 tensor ${name} byte length differs from its declared geometry.`);
  }
  const requirement = config.representationRequirement;
  if (requirement && (requirement.mode !== 'packed-only' || requirement.dtypes.some(dtype => dtype !== format.variant))) {
    throw new Error(`Q4_0 tensor ${name} cannot satisfy a different declared weight representation.`);
  }
  const buffer = acquireAlignedBuffer(size, `q4_0_${name}`);
  try {
    writeBufferAligned(getDevice(), buffer, bytes);
    return { data: createWeightBuffer(buffer, format.variant, 'row', location.shape, name), allocatedBuffers: [buffer] };
  } catch (error) { releaseBuffer(buffer); throw error; }
}
