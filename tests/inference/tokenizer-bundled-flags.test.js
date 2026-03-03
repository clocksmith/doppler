import assert from 'node:assert/strict';

const { BundledTokenizer } = await import('../../src/inference/tokenizers/bundled.js');
const { setRuntimeConfig, resetRuntimeConfig } = await import('../../src/config/runtime.js');

function createBundledTokenizer() {
  return new BundledTokenizer({
    vocabSize: 1,
    deferSpecialTokens: true,
    addBosToken: false,
    addEosToken: false,
  });
}

try {
  setRuntimeConfig({
    inference: {
      tokenizer: {
        addBosToken: true,
        addEosToken: true,
      },
    },
  });

  {
    const tokenizer = createBundledTokenizer();
    tokenizer.load({
      type: 'bpe',
      vocab: {
        hello: 0,
        '<eos>': 1,
        '<unk>': 2,
      },
      merges: [],
      specialTokens: {
        eos: 1,
        unk: 2,
      },
    });
    assert.deepEqual(tokenizer.encode('hello'), [0]);
  }

  {
    const tokenizer = createBundledTokenizer();
    tokenizer.load({
      model: {
        type: 'BPE',
        vocab: {
          hello: 0,
          '<eos>': 1,
          '<unk>': 2,
        },
        merges: [],
      },
      special_tokens_map: {
        eos_token: '<eos>',
        unk_token: '<unk>',
      },
    });
    assert.deepEqual(tokenizer.encode('hello'), [0]);
  }
} finally {
  resetRuntimeConfig();
}

console.log('tokenizer-bundled-flags.test: ok');
