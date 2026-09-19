/**
 * @param {{ decode(ids: number[], skipSpecialTokens: boolean): string }} tokenizer
 * @param {readonly number[]} generatedIds
 * @param {number} start
 * @param {readonly string[]} sequences
 */
export function matchesStopSequence(tokenizer, generatedIds, start, sequences) {
  if (sequences.length === 0) return false;
  const text = tokenizer.decode(generatedIds.slice(start), false);
  return sequences.some(sequence => text.endsWith(sequence));
}
