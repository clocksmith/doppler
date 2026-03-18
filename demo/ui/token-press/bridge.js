// =============================================================================
// Token Press — Generator Bridge
// =============================================================================
// Drives token generation step-by-step using the pipeline's public API:
//   prefillWithLogits() → decodeStepLogits() → advanceWithToken()
//
// Real logits, real sampling, real KV cache state. The press controls
// (play/pause/step/back) drive this loop — there is no separate generate()
// call.
//
// Backward stepping truncates the KV cache to the prior position.

import { getTopK } from '../../../src/inference/pipelines/text/sampling.js';
import { sample, applyRepetitionPenalty } from '../../../src/inference/pipelines/text/sampling.js';

function hasLinearAttentionState(pipeline) {
  const runtime = pipeline.linearAttentionRuntime
    ?? pipeline._state?.linearAttentionRuntime
    ?? null;
  return runtime != null && typeof runtime === 'object';
}

export function createTokenPressSession(pipeline, press, prompt, options = {}) {
  const {
    topKSize = 10,
    temperature = 0,
    topP = 1.0,
    topK: samplingTopK = 0,
    repetitionPenalty = 1.0,
    maxTokens = 256,
    useChatTemplate = false,
  } = options;

  let prefillDone = false;
  let logits = null;
  let generatedIds = [];
  let finished = false;
  let disposed = false;

  // Step-back is only safe for standard-attention models. Recurrent models
  // (linear attention, SSM) accumulate state that cannot be partially rolled
  // back without full replay.
  const hasRecurrentState = hasLinearAttentionState(pipeline);
  const supportsStepBack = !hasRecurrentState;
  const stepBackReason = hasRecurrentState
    ? 'This model uses recurrent attention (linear/SSM). Recurrent state cannot be partially rolled back.'
    : null;
  const history = [];

  const decode = (ids) => {
    try {
      return pipeline.tokenizer?.decode?.(ids, true, false) ?? `[${ids[0]}]`;
    } catch {
      return `[${ids[0]}]`;
    }
  };

  function sampleFromLogits(rawLogits) {
    const sampledLogits = Float32Array.from(rawLogits);
    applyRepetitionPenalty(sampledLogits, generatedIds, repetitionPenalty);
    const padTokenId = pipeline.tokenizer?.getSpecialTokens?.()?.pad;
    return sample(sampledLogits, {
      temperature,
      topP,
      topK: samplingTopK,
      padTokenId,
    });
  }

  async function prefill() {
    if (prefillDone || disposed) return;
    const result = await pipeline.prefillWithLogits(prompt, { useChatTemplate });
    logits = result.logits;
    generatedIds = [...(result.tokens ?? [])];
    prefillDone = true;
  }

  // stepForward: sample one token, advance KV cache, push to press.
  // Returns the token record, or null if finished.
  async function stepForward() {
    if (finished || disposed) return null;
    if (!prefillDone) await prefill();
    if (!logits) { finished = true; return null; }

    // Extract top-k from real logits (before sampling modifies them)
    const candidates = getTopK(logits, topKSize, decode);

    // Sample using the actual runtime sampling config
    const tokenId = sampleFromLogits(logits);
    const text = decode([tokenId]);
    const confidence = candidates.find(c => c.token === tokenId)?.prob ?? 0;

    // Record for backward stepping
    history.push({
      tokenId,
      seqLen: generatedIds.length,
    });

    generatedIds.push(tokenId);

    // Check stop conditions
    const eosTokens = pipeline.modelConfig?.stopTokenIds ?? [];
    if (eosTokens.includes(tokenId) || generatedIds.length >= maxTokens) {
      finished = true;
    }

    // Advance KV cache and compute next logits
    if (!finished) {
      try {
        await pipeline.advanceWithToken(tokenId);
        const result = await pipeline.decodeStepLogits([tokenId]);
        logits = result.logits;
      } catch {
        finished = true;
        logits = null;
      }
    }

    const record = { tokenId, text, topK: candidates, confidence };
    press.pushToken(record);
    return record;
  }

  // stepBack: rewind KV cache, pop the last token from press.
  // Only available for standard-attention models.
  async function stepBack() {
    if (!supportsStepBack) return null;
    if (disposed || history.length === 0) return null;

    const entry = history.pop();
    generatedIds.pop();
    finished = false;

    // Truncate KV cache to prior position
    if (pipeline.kvCache?.truncate) {
      pipeline.kvCache.truncate(entry.seqLen);
    }

    // Rewind press queue
    const removed = press.queue.stepBack();

    // Recompute logits at the new position
    try {
      const lastTokenId = generatedIds[generatedIds.length - 1];
      if (lastTokenId != null) {
        const result = await pipeline.decodeStepLogits([lastTokenId]);
        logits = result.logits;
      }
    } catch {
      logits = null;
    }

    return removed;
  }

  function dispose() {
    disposed = true;
    history.length = 0;
    logits = null;
  }

  return {
    prefill,
    stepForward,
    stepBack,
    dispose,
    get finished() { return finished; },
    get tokenCount() { return history.length; },
    get supportsStepBack() { return supportsStepBack; },
    get stepBackReason() { return stepBackReason; },
  };
}
