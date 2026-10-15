import assert from 'node:assert/strict';
import { PipelineGenerator } from '../../src/inference/pipelines/text/generator.js';
import { BundledTokenizer } from '../../src/inference/tokenizers/bundled.js';

const tokenizer = new BundledTokenizer({ vocabSize: 0, deferSpecialTokens: true, addBosToken: false, addEosToken: false });
tokenizer.load({ model: { type: 'BPE', vocab: {
  ...Object.fromEntries(Array.from({ length: 256 }, (_, i) => [`<0x${i.toString(16).padStart(2, '0')}>`, i])),
  '<eos>': 256,
}, merges: [], byte_fallback: true }, pre_tokenizer: { type: 'ByteLevel', add_prefix_space: false },
added_tokens: [{ id: 256, content: '<eos>', special: true }] });
const bytes = text => [...new TextEncoder().encode(text)];
const pipeline = Object.create(PipelineGenerator.prototype);
pipeline._state = { isLoaded: true, tokenizer, stats: {}, modelConfig: { stopTokenIds: [256] } };
let ids = [...bytes('Hello \u{1f30d} café 世界'), 256], settled = 0;
pipeline._generateTokensInternal = async function* (prompt, options, mode) {
  try {
    for (const id of ids) {
      const text = tokenizer.decode([id], true, false);
      yield mode === 'token' ? id : text;
      options.onToken?.(id, text);
    }
  } finally { settled++; }
};
const seen = [];
const chunks = await Array.fromAsync(pipeline.generate('fixture', { onToken: (id, text) => seen.push({ id, text }) }));
assert.equal(chunks.join(''), 'Hello \u{1f30d} café 世界');
assert(!chunks.some(chunk => chunk.includes('\ufffd')));
assert.deepEqual(seen.map(item => item.id), ids);
assert.equal(seen.map(item => item.text).join(''), chunks.join(''));
assert.deepEqual(await Array.fromAsync(pipeline.generateTokens('fixture')), ids);
ids = [0xf0, 0x9f];
assert.equal((await Array.fromAsync(pipeline.generate('truncated'))).join(''), '\ufffd');
ids = bytes('OK');
assert.equal((await Array.fromAsync(pipeline.generate('next'))).join(''), 'OK');
const abort = new AbortController();
ids = bytes('\u{1f30d}');
const before = settled;
await assert.rejects(() => Array.fromAsync(pipeline.generate('cancel', {
  signal: abort.signal, onToken() { abort.abort(new Error('cancel incomplete character')); },
})), /cancel incomplete character/);
assert.equal(settled, before + 1);
const stream = pipeline.generate('early return');
await stream.next();
await stream.return();
assert.equal(settled, before + 2);
console.log('generation-unicode.test: ok (public text boundary, synthetic token stream)');
