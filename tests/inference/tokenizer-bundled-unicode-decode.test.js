import assert from 'node:assert/strict';

import { BundledTokenizer } from '../../src/inference/tokenizers/bundled.js';

function createTokenizer() {
  return new BundledTokenizer({
    vocabSize: 0,
    deferSpecialTokens: true,
    addBosToken: false,
    addEosToken: false,
  });
}

{
  const tokenizer = createTokenizer();
  tokenizer.load({
    model: {
      type: 'BPE',
      vocab: {
        '<eos>': 0,
        '<unk>': 1,
        '▁reunión': 2,
        '▁mañana': 3,
        '<0xC3>': 4,
        '<0xB3>': 5,
      },
      merges: [],
      byte_fallback: true,
    },
    pre_tokenizer: {
      type: 'Split',
      pattern: { String: ' ' },
      behavior: 'MergedWithPrevious',
      invert: false,
    },
    special_tokens_map: {
      eos_token: '<eos>',
      unk_token: '<unk>',
    },
  });

  assert.equal(tokenizer.decode([2, 3]), 'reunión mañana');
  assert.equal(tokenizer.decode([4, 5]), 'ó');
}

{
  const tokenizer = createTokenizer();
  tokenizer.load({
    type: 'bpe',
    vocab: {
      '<eos>': 0,
      '<unk>': 1,
      'cafÃ©': 2,
    },
    merges: [],
    pre_tokenizer: {
      type: 'ByteLevel',
      add_prefix_space: false,
    },
    specialTokens: {
      eos: '<eos>',
      unk: '<unk>',
    },
  });

  assert.equal(tokenizer.decode([2]), 'café');
}

console.log('tokenizer-bundled-unicode-decode.test: ok');
