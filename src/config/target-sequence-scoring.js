import contract from './target-sequence-scoring.json' with { type: 'json' };
import { freezeCapsuleV2 } from "./capsule/capsule-v2.js";

export const TARGET_SEQUENCE_SCORING_CONTRACT = freezeCapsuleV2(contract);

const ROUNDING_SCALE = Math.round(1 / contract.roundingStep);

/** Sum in index order, so every consumer reproduces the same float64 total.
 * @param {readonly number[]} tokenLogProbs */
export function sumTargetSequenceLogProbs(tokenLogProbs) {
  let total = 0;
  for (const value of tokenLogProbs) total += value;
  return total;
}

/** Round half toward +Infinity to the contract step (0.01); -0 normalizes to 0.
 * @param {number} totalLogProb */
export function roundTargetSequenceLogProb(totalLogProb) {
  return (Math.round(totalLogProb * ROUNDING_SCALE) / ROUNDING_SCALE) || 0;
}

/** @param {unknown} input
 * @returns {Readonly<import('./target-sequence-scoring.js').TargetSequenceScoringRequest>} */
export function snapshotTargetSequenceScoringRequest(input) {
  const request = /** @type {Record<string, unknown> | null} */ (input);
  if (!request || typeof request !== 'object' || Array.isArray(request)
    || Object.keys(request).some(key => !['prompt', 'messages', 'target', 'maxSeqLen'].includes(key))) {
    throw new Error('Target sequence scoring accepts only prompt or messages, target and maxSeqLen.');
  }
  if ((request.prompt === undefined) === (request.messages === undefined)) {
    throw new Error('Target sequence scoring requires exactly one of prompt or messages.');
  }
  if (request.prompt !== undefined && (typeof request.prompt !== 'string' || !request.prompt.trim())) {
    throw new Error('Target sequence scoring prompt must be a non-empty, already formatted string.');
  }
  /** @type {{ role: 'system' | 'user'; content: string }[] | undefined} */
  let messages;
  if (request.messages !== undefined) {
    if (!Array.isArray(request.messages) || request.messages.length < 1) {
      throw new Error('Target sequence scoring messages must be a non-empty array.');
    }
    messages = request.messages.map((value, index) => {
      const message = /** @type {Record<string, unknown> | null} */ (value);
      if (!message || typeof message !== 'object' || Array.isArray(message)
        || Object.keys(message).some(key => !['role', 'content'].includes(key))
        || typeof message.role !== 'string' || !contract.messageRoles.includes(message.role)
        || typeof message.content !== 'string') {
        throw new Error(`Target sequence scoring message ${index} must be { role: ${contract.messageRoles
          .map(role => `"${role}"`).join(' | ')}, content: string }.`);
      }
      return { role: /** @type {'system' | 'user'} */ (message.role), content: message.content };
    });
    if (!messages.some(message => message.role === 'user' && message.content.trim())) {
      throw new Error('Target sequence scoring messages require a non-empty user message.');
    }
  }
  if (typeof request.target !== 'string' || request.target.length < 1) {
    throw new Error('Target sequence scoring requires a non-empty target string.');
  }
  if (typeof request.maxSeqLen !== 'number' || !Number.isSafeInteger(request.maxSeqLen) || request.maxSeqLen < 2) {
    throw new Error('Target sequence scoring requires maxSeqLen >= 2.');
  }
  return freezeCapsuleV2(messages
    ? { messages, target: request.target, maxSeqLen: request.maxSeqLen }
    : { prompt: /** @type {string} */ (request.prompt), target: request.target, maxSeqLen: request.maxSeqLen });
}

/** @param {import('./target-sequence-scoring.js').TargetSequenceScoringRequest} input
 * @param {unknown} value
 * @returns {import('./target-sequence-scoring.js').TargetSequenceScoreResult} */
export function validateTargetSequenceScoreResult(input, value) {
  const request = snapshotTargetSequenceScoringRequest(input);
  const result = /** @type {import('./target-sequence-scoring.js').TargetSequenceScoreResult | null} */ (value);
  if (!result || result.schema !== contract.resultSchema || result.interpretation !== contract.interpretation
    || !Number.isSafeInteger(result.tokenCount) || result.tokenCount < 1 || result.tokenCount > contract.maxTargetTokens
    || !Number.isSafeInteger(result.promptTokenCount) || result.promptTokenCount < 1
    || result.promptTokenCount + result.tokenCount > request.maxSeqLen
    || !Array.isArray(result.tokenLogProbs) || result.tokenLogProbs.length !== result.tokenCount
    || !Array.isArray(result.targetTokenIds) || result.targetTokenIds.length !== result.tokenCount
    || result.targetTokenIds.some(id => !Number.isSafeInteger(id) || id < 0)
    || result.tokenLogProbs.some(value => !Number.isFinite(value) || value > 0)) {
    throw new Error('Invalid target sequence score result contract.');
  }
  const total = sumTargetSequenceLogProbs(result.tokenLogProbs);
  if (result.totalLogProb !== total || result.meanLogProb !== total / result.tokenCount
    || result.roundedTotalLogProb !== roundTargetSequenceLogProb(total)) {
    throw new Error('Target sequence score totals do not match its token log-probabilities.');
  }
  return result;
}
