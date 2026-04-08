import assert from 'node:assert/strict';

const { BundledTokenizer } = await import('../../src/inference/tokenizers/bundled.js');

const tokenizer = new BundledTokenizer({
  modelId: 'bundled-hot-token-ranking-test',
  vocabSize: 1,
  padToken: 0,
  eosToken: 1,
  bosToken: 2,
  unkToken: 3,
});

tokenizer.load({
  model: {
    type: 'BPE',
    vocab: {
      '<pad>': 0,
      '<eos>': 1,
      '<bos>': 2,
      '<unk>': 3,
      '[multimodal]': 5,
      '<unused0>': 6,
      '▁a': 496,
      '▁to': 531,
      'The': 818,
      '▁blue': 3730,
      '▁appears': 7412,
    },
    merges: [],
  },
  added_tokens: [],
});

const hotTokenIds = tokenizer.getHotTokenIds(4);

assert.ok(Array.isArray(hotTokenIds));
assert.equal(hotTokenIds.includes(5), false, 'special-like bracket token must not be treated as hot');
assert.equal(hotTokenIds.includes(6), false, 'unused angle-bracket token must not be treated as hot');
assert.equal(hotTokenIds.includes(496), true, 'common boundary token should rank as hot');
assert.equal(hotTokenIds.includes(531), true, 'common short word should rank as hot');

console.log('tokenizer-bundled-hot-token-ranking.test: ok');
