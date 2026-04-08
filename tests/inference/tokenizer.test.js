import assert from 'node:assert/strict';

import { BPETokenizer } from '../../src/inference/tokenizers/bpe.js';

// BPETokenizer uses deferSpecialTokens=true by default from runtime config,
// so vocabSize=0 and eosToken=null are accepted at construction time.
// Special tokens and vocab are populated via load().

const VOCAB = {
  'h': 0,
  'e': 1,
  'l': 2,
  'o': 3,
  'he': 4,
  'lo': 5,
  'hel': 6,
  'hello': 7,
  ' ': 8,
  'w': 9,
  'r': 10,
  'd': 11,
  'wo': 12,
  'wor': 13,
  'world': 14,
  '<eos>': 15,
  '<bos>': 16,
  '<unk>': 17,
  '<pad>': 18,
};

const MERGES = [
  'h e',    // h + e -> he
  'l o',    // l + o -> lo
  'he l',   // he + l -> hel
  'hel lo', // hel + lo -> hello
  'w o',    // w + o -> wo
  'wo r',   // wo + r -> wor
  'wor l',  // wor + l -> worl (not in vocab, so stays split)
  'wor ld', // wor + ld -> world... but ld not in vocab
];

function makeTokenizer(overrides = {}) {
  const tok = new BPETokenizer({
    vocabSize: Object.keys(VOCAB).length,
    specialTokens: {
      eos: VOCAB['<eos>'],
      bos: VOCAB['<bos>'],
      unk: VOCAB['<unk>'],
      pad: VOCAB['<pad>'],
    },
    addBosToken: false,
    addEosToken: false,
    ...overrides,
  });
  tok.load(VOCAB, MERGES);
  return tok;
}

// === Construction: load populates vocabSize ===

{
  const tok = makeTokenizer();
  assert.equal(tok.getVocabSize(), Object.keys(VOCAB).length);
}

// === encode: simple word with BPE merges ===

{
  const tok = makeTokenizer();
  const ids = tok.encode('hello');
  assert.equal(ids.length, 1);
  assert.equal(ids[0], VOCAB['hello']);
}

// === encode: word with partial merges ===

{
  const tok = makeTokenizer();
  // "helo" -> BPE applies merges in rank order:
  // 1. 'h e' (rank 0) -> ['he', 'l', 'o']
  // 2. 'l o' (rank 1) < 'he l' (rank 2) -> ['he', 'lo']
  // 3. no merge for 'he lo' -> stop
  const ids = tok.encode('helo');
  assert.deepEqual(ids, [VOCAB['he'], VOCAB['lo']]);
}

// === encode: unknown characters fall back to <unk> ===

{
  const tok = makeTokenizer();
  // 'x' is not in vocab and no BPE merge can produce it
  const ids = tok.encode('x');
  assert.deepEqual(ids, [VOCAB['<unk>']]);
}

// === encode: empty string produces no tokens ===

{
  const tok = makeTokenizer();
  const ids = tok.encode('');
  assert.deepEqual(ids, []);
}

// === encode: whitespace handling ===

{
  const tok = makeTokenizer();
  // "he lo" splits into words ["he", " ", "lo"]
  const ids = tok.encode('he lo');
  assert.ok(ids.includes(VOCAB['he']));
  assert.ok(ids.includes(VOCAB[' ']));
  assert.ok(ids.includes(VOCAB['lo']));
}

// === encode: addBosToken prepends BOS ===

{
  const tok = makeTokenizer({ addBosToken: true });
  const ids = tok.encode('hello');
  assert.equal(ids[0], VOCAB['<bos>']);
  assert.equal(ids[ids.length - 1], VOCAB['hello']);
}

// === encode: addEosToken appends EOS ===

{
  const tok = makeTokenizer({ addEosToken: true });
  const ids = tok.encode('hello');
  assert.equal(ids[ids.length - 1], VOCAB['<eos>']);
}

// === encode: both BOS and EOS ===

{
  const tok = makeTokenizer({ addBosToken: true, addEosToken: true });
  const ids = tok.encode('hello');
  assert.equal(ids[0], VOCAB['<bos>']);
  assert.equal(ids[ids.length - 1], VOCAB['<eos>']);
  assert.ok(ids.length >= 3);
}

// === decode: basic token sequence ===

{
  const tok = makeTokenizer();
  const text = tok.decode([VOCAB['hello']]);
  assert.equal(text, 'hello');
}

// === decode: multiple tokens ===

{
  const tok = makeTokenizer();
  const text = tok.decode([VOCAB['he'], VOCAB['lo']]);
  assert.equal(text, 'helo');
}

// === decode: skipSpecialTokens=true (default) removes special tokens ===

{
  const tok = makeTokenizer();
  const text = tok.decode([VOCAB['<bos>'], VOCAB['hello'], VOCAB['<eos>']]);
  assert.equal(text, 'hello');
}

// === decode: skipSpecialTokens=false preserves special tokens ===

{
  const tok = makeTokenizer();
  const text = tok.decode([VOCAB['<bos>'], VOCAB['hello'], VOCAB['<eos>']], false, false);
  assert.ok(text.includes('<bos>'));
  assert.ok(text.includes('<eos>'));
}

// === decode: Ġ replaced with space ===

{
  const extendedVocab = { ...VOCAB, 'Ġworld': 19 };
  const tok = new BPETokenizer({
    vocabSize: Object.keys(extendedVocab).length,
    specialTokens: { eos: 15, bos: 16, unk: 17, pad: 18 },
    addBosToken: false,
    addEosToken: false,
  });
  tok.load(extendedVocab, MERGES);
  const text = tok.decode([VOCAB['hello'], 19]);
  assert.equal(text, 'hello world');
}

// === decode: Ċ replaced with newline ===

{
  const extendedVocab = { ...VOCAB, 'Ċ': 19 };
  const tok = new BPETokenizer({
    vocabSize: Object.keys(extendedVocab).length,
    specialTokens: { eos: 15, bos: 16, unk: 17, pad: 18 },
    addBosToken: false,
    addEosToken: false,
  });
  tok.load(extendedVocab, MERGES);
  const text = tok.decode([VOCAB['hello'], 19], true, false);
  assert.ok(text.includes('\n'));
}

// === decode: trim=true (default) trims whitespace ===

{
  const tok = makeTokenizer();
  const text = tok.decode([VOCAB[' '], VOCAB['hello'], VOCAB[' ']], true, true);
  assert.equal(text, 'hello');
}

// === decode: trim=false preserves surrounding whitespace ===

{
  const tok = makeTokenizer();
  const text = tok.decode([VOCAB[' '], VOCAB['hello'], VOCAB[' ']], true, false);
  assert.equal(text, ' hello ');
}

// === decode: unknown IDs are silently skipped ===

{
  const tok = makeTokenizer();
  const text = tok.decode([999, VOCAB['hello'], 888]);
  assert.equal(text, 'hello');
}

// === decode: empty array produces empty string ===

{
  const tok = makeTokenizer();
  const text = tok.decode([]);
  assert.equal(text, '');
}

// === isSpecialToken ===

{
  const tok = makeTokenizer();
  assert.equal(tok.isSpecialToken(VOCAB['<eos>']), true);
  assert.equal(tok.isSpecialToken(VOCAB['<bos>']), true);
  assert.equal(tok.isSpecialToken(VOCAB['<unk>']), true);
  assert.equal(tok.isSpecialToken(VOCAB['<pad>']), true);
  assert.equal(tok.isSpecialToken(VOCAB['hello']), false);
  assert.equal(tok.isSpecialToken(0), false);
}

// === load: reloading replaces previous vocab ===

{
  const tok = makeTokenizer();
  assert.equal(tok.getVocabSize(), Object.keys(VOCAB).length);

  const smallVocab = { 'a': 0, 'b': 1 };
  tok.load(smallVocab, []);
  assert.equal(tok.getVocabSize(), 2);

  const ids = tok.encode('a');
  assert.deepEqual(ids, [0]);
}

// === encode: throws when unk token missing and unknown char encountered ===

{
  const tok = new BPETokenizer({
    vocabSize: Object.keys(VOCAB).length,
    specialTokens: { eos: 15, unk: null },
    addBosToken: false,
    addEosToken: false,
  });
  tok.load(VOCAB, MERGES);

  assert.throws(
    () => tok.encode('x'),
    /unk token is required/
  );
}

// === BPE merge: single character word (no merges possible) ===

{
  const tok = makeTokenizer();
  const ids = tok.encode('h');
  assert.deepEqual(ids, [VOCAB['h']]);
}

// === BPE merge: two-char merge ===

{
  const tok = makeTokenizer();
  const ids = tok.encode('he');
  assert.deepEqual(ids, [VOCAB['he']]);
}

// === encode/decode round-trip ===

{
  const tok = makeTokenizer();
  const original = 'hello';
  const ids = tok.encode(original);
  const decoded = tok.decode(ids);
  assert.equal(decoded, original);
}

console.log('tokenizer.test: ok');
