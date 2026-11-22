import {
  TARGET_SEQUENCE_SCORING_CONTRACT, roundTargetSequenceLogProb, snapshotTargetSequenceScoringRequest,
  sumTargetSequenceLogProbs,
} from '../config/target-sequence-scoring.js';
import { formatChatMessages } from './pipelines/text/generator/request/chat-format.js';

/** @param {unknown} tokens
 * @returns {tokens is ArrayLike<number>} */
function isTokenList(tokens) {
  return (Array.isArray(tokens) || ArrayBuffer.isView(tokens)) && Array.from(/** @type {ArrayLike<unknown>} */ (tokens))
    .every(token => Number.isSafeInteger(token) && /** @type {number} */ (token) >= 0);
}

/** Model-ready prompt text, formatted exactly once.
 * @param {import('./target-sequence-scoring.js').TargetSequenceScoringPipeline} pipeline
 * @param {Readonly<import('../config/target-sequence-scoring.js').TargetSequenceScoringRequest>} request */
export function resolveTargetSequencePrompt(pipeline, request) {
  if (typeof request.prompt === 'string') return request.prompt;
  const templateType = pipeline.modelConfig?.chatTemplateType ?? null;
  if (typeof templateType !== 'string' || !templateType) {
    throw new Error('Target sequence scoring messages require a model chat template; pass a formatted prompt instead.');
  }
  // formatChatMessages rejects template types it does not implement.
  const chatTemplate = /** @type {import('./pipelines/text/generator/request/chat-format.js').ChatTemplateType} */ (templateType);
  return formatChatMessages(request.messages.map(message => ({ ...message })), chatTemplate,
    { thinking: TARGET_SEQUENCE_SCORING_CONTRACT.chatTemplateThinking });
}

/** Target tokens must be exactly the continuation of the prompt's own tokenization.
 * @param {import('./target-sequence-scoring.js').TargetSequenceScoringPipeline} pipeline
 * @param {string} prompt
 * @param {string} target
 * @param {number} maxSeqLen */
export function tokenizeTargetContinuation(pipeline, prompt, target, maxSeqLen) {
  const encodedPrompt = pipeline.tokenizer.encode(prompt);
  const encodedCombined = pipeline.tokenizer.encode(prompt + target);
  if (!isTokenList(encodedPrompt) || !isTokenList(encodedCombined)) {
    throw new Error('Target sequence scoring tokenizer returned invalid token ids.');
  }
  const promptTokens = Array.from(encodedPrompt);
  const combined = Array.from(encodedCombined);
  if (!promptTokens.length) throw new Error('Target sequence scoring prompt tokenized to no tokens.');
  const boundary = promptTokens.findIndex((token, index) => token !== combined[index]);
  if (boundary !== -1 || combined.length <= promptTokens.length) {
    // Typical cause: the target's first characters merge with the prompt's last token.
    throw new Error('Target sequence scoring target is not a clean token continuation of the prompt: '
      + (boundary === -1 ? 'it adds no tokens.' : `the prompt's tokenization changes at token ${boundary} `
        + 'when the target is appended (the target merges across the prompt boundary).'));
  }
  const targetTokens = combined.slice(promptTokens.length);
  if (targetTokens.length > TARGET_SEQUENCE_SCORING_CONTRACT.maxTargetTokens) {
    throw new Error(`Target sequence scoring target has ${targetTokens.length} tokens; `
      + `the contract allows at most ${TARGET_SEQUENCE_SCORING_CONTRACT.maxTargetTokens}.`);
  }
  if (combined.length > maxSeqLen) {
    throw new Error(`Target sequence scoring needs ${combined.length} tokens, which exceeds maxSeqLen ${maxSeqLen}.`);
  }
  return { promptTokens, targetTokens };
}

// Teacher-forced closeness score: log P(target | prompt) from one prefill. It consumes
// logits from Doppler's kernels and never generates or modifies a tensor.
/** @param {import('./target-sequence-scoring.js').TargetSequenceScoringPipeline} pipeline
 * @param {import('../config/target-sequence-scoring.js').TargetSequenceScoringRequest} input
 * @param {{ signal?: AbortSignal }} [control]
 * @returns {Promise<import('../config/target-sequence-scoring.js').TargetSequenceScoreResult>} */
export async function scoreTargetSequence(pipeline, input, { signal } = {}) {
  const request = snapshotTargetSequenceScoringRequest(input);
  signal?.throwIfAborted();
  if (typeof pipeline?.tokenizer?.encode !== 'function'
    || typeof pipeline.prefillWithTargetTokenLogProbs !== 'function'
    || typeof pipeline.resetGenerationState !== 'function') {
    throw new Error('Target sequence scoring requires a tokenizer and per-position target log-probabilities.');
  }
  const prompt = resolveTargetSequencePrompt(pipeline, request);
  const { promptTokens, targetTokens } = tokenizeTargetContinuation(pipeline, prompt, request.target, request.maxSeqLen);
  signal?.throwIfAborted();
  let failed = false;
  /** @type {unknown} */
  let failure;
  try {
    const result = await pipeline.prefillWithTargetTokenLogProbs(prompt, targetTokens,
      { inputIds: promptTokens, useChatTemplate: false, maxSeqLen: request.maxSeqLen, signal });
    signal?.throwIfAborted();
    if (result?.tokens?.length !== promptTokens.length
      || promptTokens.some((token, index) => token !== result.tokens[index])
      || result.targetTokenIds?.length !== targetTokens.length
      || targetTokens.some((token, index) => token !== result.targetTokenIds[index])
      || result.logProbs?.length !== targetTokens.length) {
      throw new Error('Target sequence scoring returned mismatched tokens or log-probabilities.');
    }
    const tokenLogProbs = Array.from(result.logProbs, Number);
    if (tokenLogProbs.some(value => !Number.isFinite(value) || value > 0)) {
      throw new Error('Target sequence scoring returned a non-finite or positive log-probability.');
    }
    const totalLogProb = sumTargetSequenceLogProbs(tokenLogProbs);
    return {
      schema: TARGET_SEQUENCE_SCORING_CONTRACT.resultSchema,
      interpretation: TARGET_SEQUENCE_SCORING_CONTRACT.interpretation,
      tokenCount: targetTokens.length,
      promptTokenCount: promptTokens.length,
      targetTokenIds: targetTokens,
      tokenLogProbs,
      totalLogProb,
      meanLogProb: totalLogProb / targetTokens.length,
      roundedTotalLogProb: roundTargetSequenceLogProb(totalLogProb),
    };
  } catch (error) {
    failed = true;
    failure = error;
    throw error;
  } finally {
    // Submitted work settles inside prefill before its promise resolves/rejects.
    // Only this operation's continuation is discarded; resident weights stay loaded.
    try { pipeline.resetGenerationState(); }
    catch (error) {
      if (failed) throw new AggregateError([failure, error], 'Target sequence scoring and continuation cleanup failed.');
      throw error;
    }
  }
}
