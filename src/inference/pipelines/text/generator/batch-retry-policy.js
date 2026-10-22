// A failed batch may have submitted GPU work before failing. Re-decoding the same
// position rewrites attention KV, but recurrent and convolution state cannot be
// rewound, so an in-place retry would reuse state the failed batch advanced.

/** @type {import('./batch-retry-policy.js').hasUnrewindableDecodeState} */
export function hasUnrewindableDecodeState(state) {
  return (state?.linearAttentionRuntime?.layers?.size ?? 0) > 0 || (state?.convLayerStates?.size ?? 0) > 0;
}

/** @type {import('./batch-retry-policy.js').assertBatchRetryInPlaceAllowed} */
export function assertBatchRetryInPlaceAllowed(state, error) {
  const advanced = typeof error === 'object' && error !== null && Reflect.get(error, 'decodeStateAdvanced') === true;
  if (advanced && hasUnrewindableDecodeState(state)) {
    throw new Error(
      '[Pipeline] Batch decode failed after GPU submission; recurrent decode state already advanced, '
      + 'so this position cannot be decoded again in place.',
      { cause: error }
    );
  }
}
