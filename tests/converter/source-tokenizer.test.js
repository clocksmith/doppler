import assert from 'node:assert/strict';
import {
  buildCharacterTokenizerJson,
  buildSourceTokenizerJson,
  validateSourceTokenizerPolicy,
} from '../../src/converter/source-tokenizer.js';

const policy = {
  kind: 'character_vocab',
  vocabFile: 'vocab.txt',
  specialTokens: {
    pad: '<pad>',
    bos: '<cls>',
    eos: '<eos>',
    unk: '<unk>',
    mask: '<mask>',
  },
  addBosToken: true,
  addEosToken: true,
};

assert.deepEqual(validateSourceTokenizerPolicy(policy), policy);

const tokenizer = buildCharacterTokenizerJson(
  policy,
  '<cls>\n<pad>\n<eos>\n<unk>\nA\nC\n<mask>\n'
);
assert.equal(tokenizer.model.type, 'WordPiece');
assert.equal(tokenizer.model.vocab.A, 4);
assert.equal(tokenizer.model.vocab.C, 5);
assert.deepEqual(tokenizer.pre_tokenizer, {
  type: 'Split',
  pattern: { String: '' },
  behavior: 'Removed',
  invert: false,
});
assert.equal(tokenizer.add_bos_token, true);
assert.equal(tokenizer.add_eos_token, true);
assert.equal(tokenizer.source_vocab_size, 7);
assert.deepEqual(
  tokenizer.added_tokens.map(({ id, content }) => ({ id, content })),
  [
    { id: 1, content: '<pad>' },
    { id: 0, content: '<cls>' },
    { id: 2, content: '<eos>' },
    { id: 3, content: '<unk>' },
    { id: 6, content: '<mask>' },
  ]
);

assert.throws(
  () => validateSourceTokenizerPolicy({ ...policy, vocabFile: '../vocab.txt' }),
  /must stay inside/
);
assert.throws(
  () => buildCharacterTokenizerJson(policy, '<cls>\n<pad>\n<eos>\n<unk>\nA\n'),
  /missing mask token/
);
assert.throws(
  () => buildCharacterTokenizerJson(policy, '<cls>\n<pad>\n<eos>\n<unk>\nA\nA\n<mask>\n'),
  /duplicate token/
);

const greedyPolicy = {
  kind: 'greedy_vocab',
  vocabFile: 'vocab.txt',
  specialTokens: {
    pad: '<pad>',
    bos: '<cls>',
    eos: null,
    unk: '<unk>',
    mask: '<mask>',
  },
  addBosToken: true,
  addEosToken: false,
};
assert.deepEqual(validateSourceTokenizerPolicy(greedyPolicy), greedyPolicy);
const greedyTokenizer = buildSourceTokenizerJson(
  greedyPolicy,
  '<unk>\n<pad>\n<mask>\n<cls>\nAAAAAA\nA\nT\nC\nG\nN\n'
);
assert.equal(greedyTokenizer.model.type, 'BPE');
assert.deepEqual(greedyTokenizer.model.merges, []);
assert.equal(greedyTokenizer.pre_tokenizer, null);
assert.equal(greedyTokenizer.add_bos_token, true);
assert.equal(greedyTokenizer.add_eos_token, false);
assert.equal(greedyTokenizer.special_tokens_map.eos_token, undefined);

console.log('source-tokenizer.test: ok');
