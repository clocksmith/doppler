import assert from 'node:assert/strict';

import { resolveBundledTokenizerVocabSize } from '../../src/converter/core.js';

{
  const tokenizerJson = {
    model: {
      vocab: {
        '<pad>': 0,
        '<eos>': 1,
        length: 3119,
        answer: 3120,
      },
    },
  };
  assert.equal(
    resolveBundledTokenizerVocabSize(tokenizerJson),
    4,
    'object vocab must count keys, not read a token named "length" as vocab size'
  );
}

{
  const tokenizerJson = {
    model: {
      vocab: [
        ['<pad>', 0],
        ['<eos>', 1],
        ['answer', 2],
      ],
    },
  };
  assert.equal(resolveBundledTokenizerVocabSize(tokenizerJson), 3);
}

{
  assert.equal(resolveBundledTokenizerVocabSize(null), 0);
  assert.equal(resolveBundledTokenizerVocabSize({ model: {} }), 0);
}

console.log('bundled-tokenizer-vocab-size.test: ok');
