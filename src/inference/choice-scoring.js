import { CHOICE_SCORING_CONTRACT, snapshotChoiceScoringRequest } from '../config/choice-scoring.js';

// Answer labels are validated in context, not by encoding each label in isolation.
// This operation consumes logits; it never generates prose or modifies a tensor.
/** @param {import('./choice-scoring.js').ChoiceScoringPipeline} pipeline
 * @param {import('../config/choice-scoring.js').ChoiceScoringRequest} input
 * @param {{ signal?: AbortSignal }} [control]
 * @returns {Promise<import('../config/choice-scoring.js').ChoiceScoringResult>} */
export async function scoreModelChoices(pipeline, input, { signal } = {}) {
  const request = snapshotChoiceScoringRequest(input);
  signal?.throwIfAborted();
  if (typeof pipeline?.tokenizer?.encode !== 'function'
    || typeof pipeline.prefillWithTokenLogits !== 'function'
    || typeof pipeline.resetGenerationState !== 'function') {
    throw new Error('Choice scoring requires a tokenizer and selected-token logits.');
  }
  const tokens = Array.from(pipeline.tokenizer.encode(request.prompt));
  if (!tokens.length || tokens.some(token => !Number.isSafeInteger(token) || token < 0)
    || tokens.length > request.maxSeqLen) {
    throw new Error('Choice scoring prompt is empty, invalid or exceeds maxSeqLen.');
  }
  const tokenIds = request.choices.map(choice => {
    const combined = Array.from(pipeline.tokenizer.encode(request.prompt + choice.label));
    const answerToken = combined.at(-1);
    if (combined.length !== tokens.length + 1 || tokens.some((token, index) => token !== combined[index])
      || answerToken === undefined || !Number.isSafeInteger(answerToken) || answerToken < 0) {
      throw new Error(`Choice label "${choice.id}" must append exactly one token at the answer position.`);
    }
    return answerToken;
  });
  if (new Set(tokenIds).size !== tokenIds.length) {
    throw new Error('Choice labels must resolve to distinct answer tokens.');
  }
  signal?.throwIfAborted();
  let failed = false;
  /** @type {unknown} */
  let failure;
  try {
    const result = await pipeline.prefillWithTokenLogits(request.prompt, tokenIds,
      { inputIds: tokens, useChatTemplate: false, maxSeqLen: request.maxSeqLen, signal });
    signal?.throwIfAborted();
    if (result?.logits?.length !== tokenIds.length
      || result.tokens?.length !== tokens.length
      || tokens.some((token, index) => token !== result.tokens[index])) {
      throw new Error('Choice scoring returned mismatched logits or prompt tokens.');
    }
    const choices = request.choices.map((choice, index) => {
      const logit = Number(result.logits[index]);
      if (!Number.isFinite(logit)) throw new Error('Choice scoring returned a non-finite logit.');
      return { ...choice, tokenId: tokenIds[index], logit };
    });
    // Stable first-choice tie breaking is scalar answer selection after readback.
    const selected = choices.reduce((best, choice) => choice.logit > best.logit ? choice : best);
    return { schema: CHOICE_SCORING_CONTRACT.resultSchema,
      interpretation: CHOICE_SCORING_CONTRACT.interpretation, calibration: CHOICE_SCORING_CONTRACT.calibration,
      choices, selectedId: selected.id, promptTokenCount: tokens.length };
  } catch (error) {
    failed = true;
    failure = error;
    throw error;
  } finally {
    // Submitted work settles inside prefill before its promise resolves/rejects.
    // Only this operation's continuation is discarded; resident weights stay loaded.
    try { pipeline.resetGenerationState(); }
    catch (error) {
      if (failed) throw new AggregateError([failure, error], 'Choice scoring and continuation cleanup failed.');
      throw error;
    }
  }
}
