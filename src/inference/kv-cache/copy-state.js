import { getDevice } from '../../gpu/device.js';

/** Full storage copy, including overwritten ring slots. The caller pins the
 * settled source and owns both charged caches until submission settles.
 * @type {import('./copy-state.js').copyKVCacheState} */
export async function copyKVCacheState(target, source) {
  if (!target.useGPU || !source.useGPU || target.constructor !== source.constructor
    || target.numLayers !== source.numLayers || target.maxSeqLen !== source.maxSeqLen
    || target.kvDtype !== source.kvDtype || target.layout !== source.layout
    || target.kvSize !== source.kvSize || target.pageSize !== source.pageSize
    || target.windowSize !== source.windowSize) throw new Error('KV copy requires identical GPU cache geometry.');
  const device = getDevice();
  if (!device) throw new Error('KV copy requires an initialized device.');
  const encoder = device.createCommandEncoder({ label: 'resident_kv_copy' });
  for (let i = 0; i < target.numLayers; i++) {
    const srcLayer = source.layers[i], dstLayer = target.layers[i];
    for (const field of /** @type {const} */ (['keysGPU', 'valuesGPU', 'pageTableGPU'])) {
      const src = /** @type {Partial<Record<typeof field, GPUBuffer | null>>} */ (srcLayer)[field];
      const dst = /** @type {Partial<Record<typeof field, GPUBuffer | null>>} */ (dstLayer)[field];
      if (!src && !dst) continue;
      if (!src || !dst || src.size !== dst.size || src === dst) throw new Error('KV copy buffer geometry or ownership mismatch.');
      encoder.copyBufferToBuffer(src, 0, dst, 0, src.size);
    }
    if ('pageTable' in srcLayer && srcLayer.pageTable
      && (!('pageTable' in dstLayer) || !dstLayer.pageTable || dstLayer.pageTable.length !== srcLayer.pageTable.length)) {
      throw new Error('KV copy page-table geometry mismatch.');
    }
  }
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  target.currentSeqLen = source.currentSeqLen;
  target.totalTokensSeen = source.totalTokensSeen;
  for (let i = 0; i < target.numLayers; i++) {
    const src = source.layers[i], dst = target.layers[i];
    dst.seqLen = src.seqLen;
    if ('pageTable' in src && src.pageTable && 'pageTable' in dst) dst.pageTable?.set(src.pageTable);
    if ('allocatedPages' in src && 'allocatedPages' in dst) dst.allocatedPages = src.allocatedPages;
  }
}
