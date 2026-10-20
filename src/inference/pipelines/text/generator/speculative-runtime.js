import { checkpointLinearAttentionRuntime } from '../linear-attention/state-checkpoint.js';

/** @type {import('./speculative-runtime.js').lookupDraft} */
export function lookupDraft(context, maxTokens, maxMatchTokens, minMatchTokens) {
  if (!Number.isSafeInteger(maxTokens) || maxTokens < 1 || !Number.isSafeInteger(maxMatchTokens) || maxMatchTokens < 1
    || !Number.isSafeInteger(minMatchTokens) || minMatchTokens < 1 || minMatchTokens > maxMatchTokens) {
    throw new Error('Draft lookup requires positive token and match bounds.');
  }
  for (let length = Math.min(maxMatchTokens, context.length - 1); length >= minMatchTokens; length--) {
    const tail = context.length - length;
    for (let start = tail - 1; start >= 0; start--) {
      let matched = true;
      for (let j = 0; j < length; j++) if (context[start + j] !== context[tail + j]) { matched = false; break; }
      if (matched) return context.slice(start + length, Math.min(context.length, start + length + maxTokens));
    }
  }
  return [];
}


/** @type {import('./speculative-runtime.js').assertSpeculativeCapability} */
export function assertSpeculativeCapability(state, options) {
  const config = state.modelConfig, cache = state.kvCache;
  if (!config || !cache || !state.linearAttentionRuntime || cache.layout !== 'contiguous' || state.convLayerStates.size
    || config.useMoE || config.numKvSharedLayers > 0 || state.modelPartition
    || options.temperature !== 0 || typeof options.logitMaskFn === 'function') {
    throw new Error('Draft verification requires greedy whole-model execution with contiguous state and no shared KV or independent convolution.');
  }
}

/** Target verification owns tentative recurrent state until acceptance settles.
 * KV rollback is restricted to contiguous caches: no evicted prefix can be invented.
 * @type {import('./speculative-runtime.js').verifySpeculativeTokens} */
export async function verifySpeculativeTokens(generator, context, proposal, options, signal, shouldStop) {
  const state = generator._state, cache = state.kvCache, config = state.modelConfig;
  signal?.throwIfAborted();
  assertSpeculativeCapability(state, options);
  if (!config || !cache || !state.linearAttentionRuntime) throw new Error('Target state is not loaded.');
  if (!proposal.length || context.length !== state.currentSeqLen + 1
    || proposal.some(token => !Number.isSafeInteger(token) || token < 0 || token >= config.vocabSize)) {
    throw new Error('Draft proposal does not match the target cursor or vocabulary.');
  }
  const inputs = [context[context.length - 1], ...proposal];
  if (state.currentSeqLen + inputs.length > cache.maxSeqLen) {
    throw new Error('Draft verification exceeds the configured contiguous cache capacity.');
  }
  const sequence = state.currentSeqLen;
  const snapshot = await checkpointLinearAttentionRuntime(state.linearAttentionRuntime);
  const restore = async () => {
    await snapshot.restore();
    cache.truncate(sequence); state.currentSeqLen = sequence;
  };
  try {
    signal?.throwIfAborted();
    const rows = await generator._prefill(inputs, { ...options, _returnAllLogits: true });
    signal?.throwIfAborted();
    if (!(rows instanceof Float32Array) || rows.length !== inputs.length * config.vocabSize) {
      throw new Error('Target verification requires one full logit row per proposed input.');
    }
    const tokens = [], history = [...context];
    let accepted = 0;
    for (let row = 0; row < inputs.length; row++) {
      const token = generator._sampleNextTokenFromLogits(rows.subarray(row * config.vocabSize, (row + 1) * config.vocabSize), history, options);
      history.push(token); tokens.push(token);
      if (row < proposal.length && token === proposal[row]) accepted++;
      if (shouldStop(history, token) || row === proposal.length || token !== proposal[row]) break;
    }
    if (tokens.length < inputs.length) {
      await restore();
      // Replay only the accepted prefix; rejected recurrent updates are never published.
      await generator._prefill(inputs.slice(0, tokens.length), options);
    }
    signal?.throwIfAborted();
    return { tokens, accepted };
  } catch (error) {
    try { await restore(); }
    catch (settlement) { throw new AggregateError([error, settlement], 'Draft verification and state recovery failed.'); }
    throw error;
  } finally {
    snapshot.close();
  }
}
