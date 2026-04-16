import assert from 'node:assert/strict';

const { BPETokenizer } = await import('../../src/inference/tokenizers/bpe.js');
const { BundledTokenizer } = await import('../../src/inference/tokenizers/bundled.js');
const { SentencePieceTokenizer } = await import('../../src/inference/tokenizers/sentencepiece.js');

const textEncoder = new TextEncoder();

function encodeVarint(value) {
  const out = [];
  let next = Number(value);
  do {
    let byte = next & 0x7f;
    next >>>= 7;
    if (next) byte |= 0x80;
    out.push(byte);
  } while (next);
  return out;
}

function concatBytes(chunks) {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

function encodePiece(piece, score = 0, type = 1) {
  const pieceBytes = textEncoder.encode(piece);
  const scoreBytes = new Uint8Array(4);
  new DataView(scoreBytes.buffer).setFloat32(0, score, true);
  return concatBytes([
    Uint8Array.from([0x0a, ...encodeVarint(pieceBytes.length)]),
    pieceBytes,
    Uint8Array.from([0x15]),
    scoreBytes,
    Uint8Array.from([0x18, ...encodeVarint(type)]),
  ]);
}

function encodeLiteRTPieceList(pieces) {
  return concatBytes(pieces.map((piece) => {
    const payload = encodePiece(piece.piece, piece.score, piece.type);
    return concatBytes([
      Uint8Array.from([0x0a, ...encodeVarint(payload.length)]),
      payload,
    ]);
  })).buffer;
}

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

{
  const tokenizer = new SentencePieceTokenizer({
    vocabSize: 0,
    deferSpecialTokens: true,
    specialTokens: { pad: 0, eos: 1, bos: 2, unk: 3 },
    addBosToken: false,
    addEosToken: false,
  });
  await tokenizer.load(encodeLiteRTPieceList([
    { piece: '<pad>', score: 0, type: 3 },
    { piece: '<eos>', score: 0, type: 3 },
    { piece: '<bos>', score: 0, type: 3 },
    { piece: '<unk>', score: 0, type: 2 },
    { piece: 'dup', score: 4, type: 1 },
    { piece: 'dup', score: 3, type: 1 },
    { piece: '', score: 2, type: 1 },
    { piece: '▁', score: -100, type: 1 },
    { piece: 'H', score: -100, type: 1 },
    { piece: 'e', score: -100, type: 1 },
    { piece: 'l', score: -100, type: 1 },
    { piece: 'o', score: -100, type: 1 },
    { piece: 'He', score: -1, type: 1 },
    { piece: 'll', score: -2, type: 1 },
    { piece: 'lo', score: -3, type: 1 },
    { piece: 'llo', score: -4, type: 1 },
    { piece: 'Hello', score: -5, type: 1 },
    { piece: '▁Hello', score: -6, type: 1 },
  ]));

  assert.equal(tokenizer.getVocabSize(), 18);
  assert.deepEqual(tokenizer.encode('Hello'), [16]);
  assert.deepEqual(tokenizer.encode(' Hello'), [17]);
  assert.equal(tokenizer.decode([4], false, false), 'dup');
  assert.equal(tokenizer.decode([5], false, false), 'dup');
  assert.equal(tokenizer.decode([6], false, false), '');
  assert.equal(tokenizer.decode([16]), 'Hello');
}

console.log('tokenizer-reload-cleanup.test: ok');
