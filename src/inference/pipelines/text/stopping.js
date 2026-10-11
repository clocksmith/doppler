import { decodeGenerationOutput } from '../../generation-output.js';

/**
 * @param {{ decode(ids: number[], skipSpecialTokens: boolean): string }} tokenizer
 * @param {readonly number[]} generatedIds
 * @param {number} start
 * @param {readonly string[]} sequences
 * @param {import('../../../config/reasoning-output.js').ReasoningOutputContract | null} [reasoningOutput]
 */
export function matchesStopSequence(tokenizer, generatedIds, start, sequences, reasoningOutput = null) {
  if (sequences.length === 0) return false;
  const ids = generatedIds.slice(start);
  const text = reasoningOutput == null ? tokenizer.decode(ids, false)
    : decodeGenerationOutput(/** @type {import('../../generation-output.js').OutputTokenizer} */ (tokenizer),
      reasoningOutput, [], ids, false);
  return sequences.some(sequence => text.endsWith(sequence));
}
