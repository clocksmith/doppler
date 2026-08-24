import { releaseBuffer } from '../../../../memory/buffer-pool.js';

export function resolveEffectivePrefillTokenChunkSize(state) {
  const runtimeSession = state?.runtimeConfig?.inference?.session;
  const runtimeChunkSize = runtimeSession?.prefillTokenChunkSize;
  if (runtimeChunkSize !== undefined && runtimeChunkSize !== null) {
    return runtimeChunkSize;
  }
  const modelSession = state?.modelConfig?.sessionSettings;
  if (modelSession?.prefillTokenChunkSize !== undefined) {
    return modelSession.prefillTokenChunkSize;
  }
  return runtimeChunkSize;
}

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

export function shouldDisablePrefillCommandBatching(state, opts, multimodalBidirectionalSpan) {
  if (
    opts?.disableCommandBatching === true
    || opts?.debug === true
    || (Array.isArray(opts?.debugLayers) && opts.debugLayers.length > 0)
  ) {
    return true;
  }
  if (state?.kvCache?.layout === 'bdpa_paged') {
    return true;
  }
  if (resolveEffectivePrefillTokenChunkSize(state) != null) {
    return true;
  }
  if (multimodalBidirectionalSpan == null) {
    return false;
  }
  if (state?.kvCache?.hasGPUCache?.() !== true) {
    return false;
  }
  return true;
}
