// =============================================================================
// Token Press — Generator Bridge
// =============================================================================
// Connects a Doppler InferencePipeline's generate() call to the token press
// visualization. Extracts top-k probability data per token by wrapping the
// pipeline's decodeStepLogits / CPU sampling path.
//
// Two integration modes:
//
// 1. Lightweight (post-hoc top-k):
//    Wraps the standard generate() loop. After each token is sampled, reads
//    back logits from the pipeline and computes top-k. Simple but adds one
//    readback per token.
//
// 2. Step-by-step (full control):
//    Uses prefillWithLogits() + decodeStepLogits() for manual token-by-token
//    generation. The press controls (play/pause/step/back) drive the loop.
//    No extra readback — logits are already on CPU.
//
// Usage:
//   import { runGenerationWithPress } from './token-press/bridge.js';
//
//   await runGenerationWithPress(pipeline, press, prompt, {
//     ...generateOptions,
//     signal: abortController.signal,
//   });

import { getTopK, softmax } from '../../../src/inference/pipelines/text/sampling.js';

// Lightweight mode: wrap standard generate() and extract top-k per token.
// Works with any pipeline.generate() call.
export async function runGenerationWithPress(pipeline, press, prompt, options = {}) {
  const { topKSize = 10, signal, ...generateOptions } = options;

  const decode = (ids) => {
    try {
      return pipeline.tokenizer?.decode?.(ids, true, false) ?? `[${ids[0]}]`;
    } catch {
      return `[${ids[0]}]`;
    }
  };

  let tokenIndex = 0;

  for await (const tokenText of pipeline.generate(prompt, {
    ...generateOptions,
    signal,
  })) {
    if (signal?.aborted) break;

    // The pipeline has already sampled this token. We can get the logits
    // from the most recent decode step if the pipeline exposes them.
    // For now, create a record with the token text and confidence from
    // the pipeline's last sampling stats if available.
    const stats = pipeline.getStats?.() ?? {};
    const lastTopK = stats.lastTopK ?? null;

    let topK;
    let confidence;

    if (lastTopK && lastTopK.length > 0) {
      topK = lastTopK.slice(0, topKSize);
      confidence = topK[0]?.prob ?? 0;
    } else {
      // Fallback: we don't have logits, just record the token
      topK = [{ token: -1, logit: 0, prob: 1, text: tokenText }];
      confidence = 1;
    }

    press.pushToken({
      tokenId: topK[0]?.token ?? tokenIndex,
      text: tokenText,
      topK,
      confidence,
    });

    tokenIndex++;
  }
}

// Step-by-step mode: manual token generation with full logit access.
// Returns a controller that the press controls can drive.
export function createStepGenerator(pipeline, prompt, options = {}) {
  const { topKSize = 10, ...generateOptions } = options;

  let prefillDone = false;
  let logits = null;
  let generatedIds = [];
  let finished = false;

  const decode = (ids) => {
    try {
      return pipeline.tokenizer?.decode?.(ids, true, false) ?? `[${ids[0]}]`;
    } catch {
      return `[${ids[0]}]`;
    }
  };

  async function prefill() {
    if (prefillDone) return;
    const result = await pipeline.prefillWithLogits(prompt, generateOptions);
    logits = result.logits;
    generatedIds = [...(result.tokens ?? [])];
    prefillDone = true;
  }

  async function stepForward() {
    if (finished) return null;
    if (!prefillDone) await prefill();

    if (!logits) {
      finished = true;
      return null;
    }

    const topK = getTopK(logits, topKSize, decode);
    const confidence = topK[0]?.prob ?? 0;
    const tokenId = topK[0]?.token ?? 0;
    const text = decode([tokenId]);

    generatedIds.push(tokenId);

    // Check for EOS
    const eosTokens = pipeline.modelConfig?.stopTokenIds ?? [];
    if (eosTokens.includes(tokenId)) {
      finished = true;
    }

    // Compute next logits (for the following step)
    if (!finished) {
      try {
        const result = await pipeline.decodeStepLogits([tokenId], generateOptions);
        logits = result.logits;
      } catch {
        finished = true;
        logits = null;
      }
    }

    return {
      tokenId,
      text,
      topK,
      confidence,
    };
  }

  return {
    prefill,
    stepForward,
    get finished() { return finished; },
    get generatedIds() { return [...generatedIds]; },
  };
}
