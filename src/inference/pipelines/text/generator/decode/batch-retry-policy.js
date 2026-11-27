// A failed batch may have submitted GPU work before failing. Re-decoding the same
// position rewrites attention KV, but recurrent and convolution state cannot be
// rewound, so an in-place retry would reuse state the failed batch advanced.

/** @type {import('./batch-retry-policy.js').hasUnrewindableDecodeState} */
export function hasUnrewindableDecodeState(state) {
  return (state?.linearAttentionRuntime?.layers?.size ?? 0) > 0 || (state?.convLayerStates?.size ?? 0) > 0;
}

// A short batch that did not end on a stop token leaves the GPU N steps ahead while
// currentSeqLen advanced only actualCount; KV is rewritten by position, recurrent
// state is not. Today linear-attention models never take hot-vocabulary batches
// (execution rules), so this is unreachable; it fails closed if that changes.
/** @type {import('./batch-retry-policy.js').assertShortBatchContinuationAllowed} */
export function assertShortBatchContinuationAllowed(state, actualCount, requestedCount) {
  if (actualCount < requestedCount && hasUnrewindableDecodeState(state)) {
    throw new Error(
      `[Pipeline] Batch decode resolved ${actualCount} of ${requestedCount} tokens without a stop token; `
      + 'recurrent decode state advanced past the delivered tokens and cannot be rewound.'
    );
  }
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
