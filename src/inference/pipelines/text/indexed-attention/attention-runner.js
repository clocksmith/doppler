import { recordIndexedAttention } from '../../../../gpu/kernels/qsa-indexer.js';
import { createTensor } from '../../../../gpu/tensor.js';

// Attention-core runner for qwen4_exp QSA layers (attention kernel variant
// `qsaIndexed`). The hyper-connection layer executor computes the selection
// from the owned indexer cache before the attention call and supplies it as
// `state.qsa`; this runner attends to the selected rows of the contiguous
// KV cache, which already holds the current tokens. Main attention consumes
// the original per-token keys and values.

/** @param {number} layerIdx @param {string} message @returns {never} */
function fail(layerIdx, message) {
  throw new Error(`[QSA] layer ${layerIdx}: ${message}`);
}

/** @type {typeof import('./attention-runner.js').runQsaIndexedAttention} */
export async function runQsaIndexedAttention(recorder, options) {
  const {
    state, qTensor, cachedKTensor, cachedVTensor, cachedKDtype, cachedVDtype, kvState, numTokens, numHeads, numKVHeads,
    headDim, attnScale, attnSoftcap, effectiveSlidingWindow, causalForAttention, layerIdx,
  } = options;
  const qsa = state?.qsa ?? null;
  if (!qsa || qsa.layerIdx !== layerIdx || !qsa.selection) fail(layerIdx, 'no indexer selection was prepared for this layer.');
  if (qsa.numTokens !== numTokens) fail(layerIdx, `selection covers ${qsa.numTokens} queries but attention has ${numTokens}.`);
  if ((cachedKDtype !== 'f16' && cachedKDtype !== 'f32') || cachedKDtype !== cachedVDtype) {
    fail(layerIdx, 'indexed attention requires matching f16 or f32 key/value storage.');
  }
  if (kvState?.kvLayout !== 'contiguous' || (kvState.kvStart ?? 0) !== 0) fail(layerIdx, 'indexed attention requires the contiguous KV layout from row 0.');
  if (qTensor?.dtype !== 'f32') fail(layerIdx, 'indexed attention requires f32 queries.');
  if (attnSoftcap) fail(layerIdx, 'attention logit softcapping is not part of qwen4_exp QSA.');
  if (effectiveSlidingWindow) fail(layerIdx, 'sliding-window attention is not part of qwen4_exp QSA.');
  if (causalForAttention !== true) fail(layerIdx, 'QSA selection is causal.');
  const kvRows = kvState.kvLenForAttention;
  if (!Number.isSafeInteger(kvRows) || kvRows !== qsa.visibleRows) {
    fail(layerIdx, `KV cache holds ${kvRows} rows but the selection was made over ${qsa.visibleRows}.`);
  }
  // Head RMSNorm preserves token-major bytes but exposes one row per head.
  // Adapt only that known view (or an already canonical query) to the core's
  // token-major matrix contract, using declared geometry without a copy.
  const shape = qTensor.shape;
  if (!Array.isArray(shape) || shape.length !== 2
    || !((shape[0] === numTokens && shape[1] === numHeads * headDim)
      || (shape[0] === numTokens * numHeads && shape[1] === headDim))) {
    fail(layerIdx, 'query must preserve the declared token-major query/head geometry.');
  }
  const query = createTensor(qTensor.buffer, 'f32', [numTokens, numHeads * headDim], 'qsa_attention_query');
  return recordIndexedAttention(recorder, query, cachedKTensor.buffer, cachedVTensor.buffer, qsa,
    { numTokens, numHeads, numKVHeads, headDim, scale: attnScale, kvRows, kvDtype: cachedKDtype });
}
