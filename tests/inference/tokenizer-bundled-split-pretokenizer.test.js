import assert from 'node:assert/strict';

import { BundledTokenizer } from '../../src/inference/tokenizers/bundled.js';

const tokenizer = new BundledTokenizer({
  vocabSize: 1,
  deferSpecialTokens: true,
  specialTokens: { pad: 90, bos: 91, eos: 92, unk: 93 },
  addBosToken: false,
  addEosToken: false,
});

tokenizer.load({
  version: '1.0',
  model: {
    type: 'BPE',
    vocab: {
      'Ġ': 0,
      n: 1,
      u: 2,
      m: 3,
      'ĠĠ': 4,
      'ĠĠĠ': 5,
      'ĠĠĠĠ': 6,
      nu: 7,
      num: 8,
      'Ġnum': 9,
      '<pad>': 90,
      '<bos>': 91,
      '<eos>': 92,
      '<unk>': 93,
    },
    merges: [
      'Ġ Ġ',
      'ĠĠ ĠĠ',
      'n u',
      'nu m',
      'Ġ num',
      'ĠĠ Ġ',
    ],
  },
  pre_tokenizer: {
    type: 'Sequence',
    pretokenizers: [
      {
        type: 'Split',
        pattern: {
          Regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        },
        behavior: 'Isolated',
        invert: false,
      },
      {
        type: 'ByteLevel',
        add_prefix_space: false,
        trim_offsets: false,
        use_regex: false,
      },
    ],
  },
  added_tokens: [
    { id: 90, content: '<pad>', special: true },
    { id: 91, content: '<bos>', special: true },
    { id: 92, content: '<eos>', special: true },
    { id: 93, content: '<unk>', special: true },
  ],
});

assert.deepEqual(tokenizer.encode('    num'), [5, 9]);
assert.equal(tokenizer.decode([5, 9], false, false), '    num');

console.log('tokenizer-bundled-split-pretokenizer.test: ok');
