import assert from 'node:assert/strict';

const { BPETokenizer } = await import('../../src/inference/tokenizers/bpe.js');
const { BundledTokenizer } = await import('../../src/inference/tokenizers/bundled.js');
const { SentencePieceTokenizer } = await import('../../src/inference/tokenizers/sentencepiece.js');

{
  const tokenizer = new BPETokenizer({
    vocabSize: 0,
    deferSpecialTokens: true,
    specialTokens: { eos: 1, unk: 99 },
  });
  tokenizer.load({ hello: 0 }, []);
  assert.equal(tokenizer.getVocabSize(), 1);

  tokenizer.load({ world: 0 }, []);
  assert.equal(tokenizer.getVocabSize(), 1);
}

{
  const tokenizer = new BundledTokenizer({
    vocabSize: 0,
    deferSpecialTokens: true,
    addBosToken: false,
    addEosToken: false,
    specialTokens: { eos: 1, unk: 2 },
  });
  tokenizer.load({
    type: 'bpe',
    vocab: {
      hello: 0,
      '<eos>': 1,
      '<unk>': 2,
    },
    merges: [],
    specialTokens: { eos: '<eos>', unk: '<unk>' },
  });
  assert.equal(tokenizer.getVocabSize(), 3);

  tokenizer.load({
    type: 'bpe',
    vocab: {
      world: 0,
      '<eos>': 1,
      '<unk>': 2,
    },
    merges: [],
    specialTokens: { eos: '<eos>', unk: '<unk>' },
  });
  assert.equal(tokenizer.getVocabSize(), 3);
}

{
  const tokenizer = new SentencePieceTokenizer({
    vocabSize: 0,
    deferSpecialTokens: true,
    specialTokens: { eos: 2, unk: 0 },
  });
  const partialThenInvalid = new Uint8Array([
    0x1a, 0x05, 0x0a, 0x03, 0x66, 0x6f, 0x6f,
    0x1a,
  ]).buffer;

  await tokenizer.load(partialThenInvalid);
  assert.equal(tokenizer.getVocabSize(), 259);
}

console.log('tokenizer-reload-cleanup.test: ok');
