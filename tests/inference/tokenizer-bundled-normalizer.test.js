import assert from 'node:assert/strict';
import { BundledTokenizer } from '../../src/inference/tokenizers/bundled.js';

function load(normalizer, tokenizer = new BundledTokenizer({ vocabSize: 1, deferSpecialTokens: true, addBosToken: false, addEosToken: false })) {
  tokenizer.load({
    model: { type: 'BPE', vocab: { '<unk>': 0, '<|im_start|>': 1, '▁': 2, h: 3, i: 4, '▁h': 5, '▁hi': 6, '\n': 7, 'é': 8, e: 9, '\u0301': 10 }, merges: [['▁', 'h'], ['▁h', 'i']] },
    added_tokens: [{ id: 0, content: '<unk>', special: true }, { id: 1, content: '<|im_start|>', special: true }],
    normalizer,
    add_eos_token: false,
  });
  return tokenizer;
}

const prefix = load({ type: 'Sequence', normalizers: [{ type: 'Prepend', prepend: '▁' }, { type: 'Replace', pattern: { String: ' ' }, content: '▁' }] });
assert.deepEqual(prefix.encode('hi'), [6]);
assert.deepEqual(prefix.encode('<|im_start|>hi'), [1, 6]);
assert.deepEqual(prefix.encode('hi<|im_start|>\nhi'), [6, 1, 2, 7, 3, 4]);
assert.deepEqual(prefix.encode(' hi'), [2, 6]);
assert.deepEqual(prefix.encode(''), []);

const unicode = load({ type: 'NFC' });
assert.deepEqual(unicode.encode('e\u0301'), [8]);
assert.deepEqual(unicode.encode('<|im_start|>e\u0301'), [1, 8]);
assert.deepEqual(load(null).encode('e\u0301'), [9, 10]);
assert.deepEqual(load(null, unicode).encode('e\u0301'), [9, 10]);
assert.deepEqual(load({ type: 'Replace', pattern: { String: 'hi' }, content: 'é' }).encode('hi'), [8]);
assert.throws(() => load({ type: 'Sequence', normalizers: [null] }), /normalizer/i);
assert.throws(() => load({ type: 'Prepend', prepend: 1 }), /normalizer/i);
assert.throws(() => load({ type: 'Replace', pattern: { Regex: '.' }, content: 'x' }), /normalizer/i);
assert.throws(() => load({ type: 'UnrecognizedNormalizer' }), /normalizer/i);

console.log('tokenizer-bundled-normalizer.test: ok');
