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

  {
    const tokenizer = new BundledTokenizer({
      vocabSize: 1,
      deferSpecialTokens: true,
      addEosToken: false,
    });
    tokenizer.load({
      model: {
        type: 'BPE',
        vocab: {
          hello: 0,
        },
        merges: [],
      },
      added_tokens: [
        { id: 2, content: '<bos>', special: true },
        { id: 1, content: '<eos>', special: true },
        { id: 3, content: '<unk>', special: true },
      ],
      post_processor: {
        type: 'TemplateProcessing',
        single: [
          { SpecialToken: { id: '<bos>', type_id: 0 } },
          { Sequence: { id: 'A', type_id: 0 } },
        ],
        special_tokens: {
          '<bos>': { id: '<bos>', ids: [2], tokens: ['<bos>'] },
        },
      },
    });
    assert.deepEqual(tokenizer.encode('hello'), [2, 0]);
  }
} finally {
  resetRuntimeConfig();
}

console.log('tokenizer-bundled-flags.test: ok');
