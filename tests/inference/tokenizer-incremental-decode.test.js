import assert from 'node:assert/strict';
import { BundledTokenizer } from '../../src/inference/tokenizers/bundled.js';

const bytes = Object.fromEntries(Array.from({ length: 256 }, (_, i) => [`<0x${i.toString(16).padStart(2, '0')}>`, i]));
function create(byteLevel, wordPiece = false) {
  const tokenizer = new BundledTokenizer({ vocabSize: 0, deferSpecialTokens: true, addBosToken: false, addEosToken: false });
  tokenizer.load({ model: { type: wordPiece ? 'WordPiece' : 'BPE', vocab: { ...bytes, '▁hello': 256, 'Ġworld': 257, '##ing': 258, 'Ċ': 259, '<eos>': 260 },
    merges: [], byte_fallback: true, continuing_subword_prefix: '##' },
  pre_tokenizer: byteLevel ? { type: 'ByteLevel', add_prefix_space: false } : { type: 'Whitespace' },
  added_tokens: [{ id: 260, content: '<eos>', special: true }] });
  return tokenizer;
}
function verify(tokenizer, ids) {
  const decoder = tokenizer.createIncrementalDecoder();
  let output = '';
  for (let i = 0; i < ids.length; i++) {
    output += decoder.push(ids[i]);
    assert.equal(output + decoder.pendingText(), tokenizer.decode(ids.slice(0, i + 1), true, false), `prefix ${i}`);
  }
  output += decoder.finish();
  assert.equal(output, tokenizer.decode(ids, true, false));
  assert.throws(() => decoder.push(0), /closed/);
}
for (const byteLevel of [true, false]) {
  const tokenizer = create(byteLevel);
  for (const text of ['你好 🌍 café é\n', '\ufeffx\ufeffy', 'a\u0000b', '𐀀🧑🏽‍💻']) {
    verify(tokenizer, [...new TextEncoder().encode(text), 260]);
  }
  for (const ids of [[0xe2], [0xf0, 0x9f], [0xe0, 0x80], [0xed, 0xa0], [0xf4, 0x90], [0xc3, 256, 0xb3], [0xef, 0xbb, 0xbf, 256, 0xef, 0xbb, 0xbf]]) verify(tokenizer, ids);
  let seed = 7;
  const random = Array.from({ length: 1500 }, () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed % 261; });
  verify(tokenizer, random);
  const first = tokenizer.createIncrementalDecoder();
  const second = tokenizer.createIncrementalDecoder();
  assert.equal(first.push(0xc3), '');
  assert.equal(second.push(65), 'A');
  assert.equal(first.push(0xb3), 'ó');
}
verify(create(false, true), [256, 258, 257, 260, 259]);
console.log('tokenizer-incremental-decode.test: ok');
