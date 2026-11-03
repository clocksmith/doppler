export { releasePerLayerInputBuffer } from './per-layer-buffer.js';

/** @type {typeof import('./prefill-policy.js').resolvePrefillMaxInFlightChunks} */
export function resolvePrefillMaxInFlightChunks(state) {
  const limit = state?.runtimeConfig?.inference?.session?.prefillMaxInFlightChunks;
  if (typeof limit !== 'number' || !Number.isSafeInteger(limit) || limit <= 0) {
    throw new Error('inference.session.prefillMaxInFlightChunks is required and must be a positive safe integer.');
  }
  return limit;
}

/** @type {typeof import('./prefill-policy.js').resolveEffectivePrefillTokenChunkSize} */
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

/** @type {typeof import('./prefill-policy.js').resolvePrefillTokenChunkSize} */
export function resolvePrefillTokenChunkSize(state, numTokens) {
  const chunkSize = resolveEffectivePrefillTokenChunkSize(state);
  if (chunkSize === undefined) {
    throw new Error('inference.session.prefillTokenChunkSize is required; use null to disable token-chunked prefill.');
  }
  if (chunkSize === null) return null;
  if (!Number.isInteger(chunkSize) || chunkSize <= 0) {
    throw new Error('inference.session.prefillTokenChunkSize must be null or a positive integer.');
  }
  return chunkSize < numTokens ? chunkSize : null;
}

// call: { startPos, returnHidden } of this _prefill call. Under a token-chunk policy only
// calls that share KV with another chunk call bypass command batching: one continuing a
// sequence (startPos > 0) or a non-final chunk whose hidden output feeds the next. A prompt
// that fits one chunk runs as one direct call, the path chunked KV writes must match.
/** @type {typeof import('./prefill-policy.js').shouldDisablePrefillCommandBatching} */
export function shouldDisablePrefillCommandBatching(state, opts, multimodalBidirectionalSpan, call) {
  if (!Number.isSafeInteger(call?.startPos) || call.startPos < 0 || typeof call.returnHidden !== 'boolean') {
    throw new Error('shouldDisablePrefillCommandBatching requires the call { startPos, returnHidden }.');
  }
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
  if (resolveEffectivePrefillTokenChunkSize(state) != null && (call.startPos > 0 || call.returnHidden)) {
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
