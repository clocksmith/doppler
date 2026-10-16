import { releaseBuffer } from '../../../../memory/buffer-pool.js';

/** @type {import('./per-layer-buffer.js').releasePerLayerInputBuffer} */
export function releasePerLayerInputBuffer(buffer, recorder, decodeBuffers, pleCache = null) {
  if (!buffer) {
    return;
  }
  const ownsBuffer = decodeBuffers?.ownsBuffer(buffer) ?? false;
  if (ownsBuffer) {
    return;
  }
  const cachedPleBuffer = pleCache?.ownedBuffers instanceof Set && pleCache.ownedBuffers.has(buffer);
  if (cachedPleBuffer) {
    return;
  }
  if (recorder) {
    recorder.trackTemporaryBuffer(buffer);
    return;
  }
  releaseBuffer(buffer);
}
