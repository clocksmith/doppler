import assert from 'node:assert/strict';

import { BundledTokenizer } from '../../src/inference/tokenizers/bundled.js';

function createTokenizer() {
  return new BundledTokenizer({
    vocabSize: 1,
    deferSpecialTokens: true,
  });
}

{
  const tokenizer = createTokenizer();
  tokenizer.load({
    model: {
      type: 'WordPiece',
      vocab: {
        '<pad>': 0,
        '<unk>': 1,
        '<mask>': 2,
        '<bos>': 3,
        '<eos>': 4,
        A: 5,
        C: 6,
        D: 7,
      },
      unk_token: '<unk>',
      continuing_subword_prefix: '##',
      max_input_chars_per_word: 100,
    },
    pre_tokenizer: {
      type: 'Split',
      pattern: { String: '' },
      behavior: 'Removed',
      invert: false,
    },
    added_tokens: [
      { id: 0, content: '<pad>', special: true },
      { id: 1, content: '<unk>', special: true },
      { id: 2, content: '<mask>', special: true },
      { id: 3, content: '<bos>', special: true },
      { id: 4, content: '<eos>', special: true },
    ],
    post_processor: {
      type: 'TemplateProcessing',
      single: [
        { SpecialToken: { id: '<bos>', type_id: 0 } },
        { Sequence: { id: 'A', type_id: 0 } },
        { SpecialToken: { id: '<eos>', type_id: 0 } },
      ],
      special_tokens: {
        '<bos>': { id: '<bos>', ids: [3], tokens: ['<bos>'] },
        '<eos>': { id: '<eos>', ids: [4], tokens: ['<eos>'] },
      },
    },
  });

  assert.deepEqual(tokenizer.encode('ACD'), [3, 5, 6, 7, 4]);
  assert.deepEqual(tokenizer.encode('A<mask>D'), [3, 5, 2, 7, 4]);
  assert.equal(tokenizer.decode([3, 5, 6, 7, 4]), 'ACD');
}

{
  const tokenizer = createTokenizer();
  tokenizer.load({
    model: {
      type: 'WordPiece',
      vocab: {
        '<unk>': 0,
        '<eos>': 1,
        play: 2,
        '##ing': 3,
        ball: 4,
      },
      unk_token: '<unk>',
      continuing_subword_prefix: '##',
      max_input_chars_per_word: 100,
    },
    added_tokens: [
      { id: 0, content: '<unk>', special: true },
      { id: 1, content: '<eos>', special: true },
    ],
  });

  assert.deepEqual(tokenizer.encode('playing ball'), [2, 3, 4]);
  assert.equal(tokenizer.decode([2, 3, 4]), 'playing ball');
}

{
  const tokenizer = createTokenizer();
  tokenizer.load({
    model: {
      type: 'BPE',
      vocab: {
        '<cls>': 0,
        '<pad>': 1,
        '<eos>': 2,
        '<unk>': 3,
        A: 4,
        C: 5,
      },
      merges: [],
    },
    added_tokens: [
      { id: 0, content: '<cls>', special: true },
      { id: 1, content: '<pad>', special: true },
      { id: 2, content: '<eos>', special: true },
      { id: 3, content: '<unk>', special: true },
    ],
    pre_tokenizer: {
      type: 'Split',
      pattern: { String: '' },
      behavior: 'Removed',
      invert: false,
    },
    post_processor: {
      type: 'TemplateProcessing',
      single: [
        { SpecialToken: { id: '<cls>', type_id: 0 } },
        { Sequence: { id: 'A', type_id: 0 } },
        { SpecialToken: { id: '<eos>', type_id: 0 } },
      ],
      special_tokens: {
        '<cls>': { id: '<cls>', ids: [0], tokens: ['<cls>'] },
        '<eos>': { id: '<eos>', ids: [2], tokens: ['<eos>'] },
      },
    },
  });

  assert.deepEqual(tokenizer.encode('AC'), [0, 4, 5, 2]);
  assert.equal(tokenizer.decode([0, 4, 5, 2]), 'AC');
}

{
  const tokenizer = createTokenizer();
  tokenizer.load({
    model: {
      type: 'BPE',
      vocab: {
        '<unk>': 0,
        '<pad>': 1,
        '<mask>': 2,
        '<cls>': 3,
        ATTCCG: 4,
        A: 5,
        T: 6,
        C: 7,
        G: 8,
      },
      merges: [],
    },
    added_tokens: [
      { id: 0, content: '<unk>', special: true },
      { id: 1, content: '<pad>', special: true },
      { id: 2, content: '<mask>', special: true },
      { id: 3, content: '<cls>', special: true },
    ],
    special_tokens_map: {
      unk_token: '<unk>',
      pad_token: '<pad>',
      mask_token: '<mask>',
      bos_token: '<cls>',
    },
    add_bos_token: true,
    add_eos_token: false,
  });

  assert.deepEqual(tokenizer.encode('ATTCCGATTCCGA'), [3, 4, 4, 5]);
}

console.log('tokenizer-bundled-wordpiece.test: ok');
