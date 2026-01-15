import { describe, it, expect } from 'vitest';
import { tokenizeTextPairs } from '../../src/training/datasets/text-pairs.js';
import { buildTokenBatch } from '../../src/training/datasets/token-batch.js';

describe('training/datasets', () => {
  it('tokenizes text pairs into causal samples', async () => {
    const tokenizer = { encode: (text) => text.split('').map((ch) => ch.charCodeAt(0)) };
    const samples = await tokenizeTextPairs(tokenizer, [
      { prompt: 'hi', completion: ' there' },
    ]);
    expect(samples.length).toBe(1);
    expect(samples[0].inputIds.length).toBe(samples[0].targetIds.length);
  });

  it('builds token batch from samples', () => {
    const batch = buildTokenBatch([
      { inputIds: [1, 2], targetIds: [2, 3] },
      { inputIds: [4], targetIds: [5] },
    ]);
    expect(batch.inputFlat.length).toBe(3);
    expect(batch.targetFlat.length).toBe(3);
    expect(batch.offsets.length).toBe(2);
  });
});
