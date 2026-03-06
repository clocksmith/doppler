import assert from 'node:assert/strict';

const { BundledTokenizer } = await import('../../src/inference/tokenizers/bundled.js');

function createBundledTokenizer() {
  return new BundledTokenizer({
    vocabSize: 1,
    deferSpecialTokens: true,
    addBosToken: false,
    addEosToken: false,
    specialTokens: {
      eos: '<|im_end|>',
      unk: '<|im_end|>',
    },
  });
}

{
  const tokenizer = createBundledTokenizer();
  tokenizer.load({
    type: 'bpe',
    vocab: {
      hello: 0,
    },
    merges: [],
    added_tokens: [
      {
        id: 10,
        content: '<|im_start|>',
        special: true,
      },
      {
        id: 11,
        content: '<|im_end|>',
        special: true,
      },
      {
        id: 12,
        content: '<think>',
        special: false,
      },
    ],
  });

  assert.deepEqual(tokenizer.encode('<|im_start|><think>hello<|im_end|>'), [10, 12, 0, 11]);
  assert.equal(tokenizer.decode([10, 12, 0, 11], true, false), '<think>hello');
}

{
  const tokenizer = createBundledTokenizer();
  tokenizer.load({
    model: {
      type: 'BPE',
      vocab: {
        hello: 0,
      },
      merges: [],
    },
    added_tokens: [
      {
        id: 10,
        content: '<|im_start|>',
        special: true,
      },
      {
        id: 11,
        content: '<|im_end|>',
        special: true,
      },
      {
        id: 12,
        content: '<think>',
        special: false,
      },
    ],
    special_tokens_map: {
      eos_token: '<|im_end|>',
    },
  });

  assert.deepEqual(tokenizer.encode('<|im_start|><think>hello<|im_end|>'), [10, 12, 0, 11]);
}

console.log('tokenizer-bundled-added-tokens.test: ok');
