import { applyRepetitionPenalty, applyPresencePenalty, sample } from './token-sampling.js';

/** @type {import('./generation-step.js').sampleCapsuleLogits} */
export function sampleCapsuleLogits(sourceLogits, contextTokens, options, tokenContract = {}) {
  const logits = Float32Array.from(sourceLogits || []);
  if (logits.length === 0) throw new Error('Capsule execution returned empty logits.');
  applyRepetitionPenalty(logits, contextTokens, options.repetitionPenalty, options.repetitionPenaltyWindow);
  applyPresencePenalty(logits, contextTokens, options.presencePenalty, options.repetitionPenaltyWindow);
  return sample(logits, { ...options, padTokenId: tokenContract.padTokenId });
}

/** @type {import('./generation-step.js').stoppingReason} */
export function stoppingReason(tokenId, generatedCount, options, tokenContract, getText) {
  if (tokenId === tokenContract.eosTokenId) return 'eos-token';
  if (tokenContract.stopTokenIds?.includes(tokenId)) return 'stop-token';
  if (options.stopSequences.length > 0) {
    const text = getText();
    if (options.stopSequences.some(sequence => text.endsWith(sequence))) return 'stop-sequence';
  }
  return generatedCount >= options.maxTokens ? 'max-tokens' : null;
}

