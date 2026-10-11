import assert from 'node:assert/strict';
import { BundledTokenizer } from '../../src/inference/tokenizers/bundled.js';
import { createGenerationOutputDecoder, decodeGenerationOutput, decodeGenerationStream } from '../../src/inference/generation-output.js';
import { validateReasoningOutput } from '../../src/config/reasoning-output.js';
import { createCapsuleOperationAdapters } from '../../src/client/runtime/capsule-operation-adapters.js';
import { PipelineGenerator } from '../../src/inference/pipelines/text/generator.js';
import { matchesStopSequence } from '../../src/inference/pipelines/text/stopping.js';
import { createModelHandle } from '../../src/client/model-host/model-session.js';

const tokenizer = new BundledTokenizer({ vocabSize: 0, deferSpecialTokens: true, addBosToken: false, addEosToken: false });
const markers = ['<think>', '</think>', '<eos>'];
tokenizer.load({ model: { type: 'BPE', vocab: {
  ...Object.fromEntries(Array.from({ length: 256 }, (_, i) => [`<0x${i.toString(16).padStart(2, '0')}>`, i])),
  ...Object.fromEntries(markers.map((text, i) => [text, 256 + i])),
}, merges: [], byte_fallback: true }, pre_tokenizer: { type: 'ByteLevel', add_prefix_space: false },
added_tokens: markers.map((content, i) => ({ id: 256 + i, content, special: true })) });
const contract = { schema: 'doppler.reasoning-output/v1', openingToken: { id: 256, text: '<think>' },
  closingToken: { id: 257, text: '</think>' } };
const bytes = text => [...new TextEncoder().encode(text)];
const ids = [256, ...bytes('private 🧑🏽‍💻'), 257, ...bytes('Yes 🌍'), 258];
const decoder = () => createGenerationOutputDecoder(tokenizer, contract, [258]);
const expected = 'Yes 🌍';
assert.equal(decodeGenerationOutput(tokenizer, contract, [258], ids), expected);
assert.equal(matchesStopSequence(tokenizer, [256, ...bytes('secretSTOP')], 0, ['STOP'], contract), false);
assert.equal(matchesStopSequence(tokenizer, [256, 65, 257, ...bytes('finalSTOP')], 0, ['STOP'], contract), true);
for (const disabled of [undefined, null]) {
  assert.equal(decodeGenerationOutput(tokenizer, disabled, [258], ids), tokenizer.decode(ids, true, false));
}
const a = decoder(), b = decoder();
let output = '';
for (let i = 0; i < ids.length; i++) {
  output += a.push(ids[i]);
  assert.equal(a.getUsage().rawCharacters, tokenizer.decode(ids.slice(0, i + 1), false, false).length);
  b.push(ids[i]);
}
assert.equal(output + a.finish(), expected);
assert.equal(b.finish(), '');
assert.equal(a.getUsage().rawTokens, ids.length);
assert.equal(a.getUsage().finalCharacters, expected.length);
assert.equal(a.getUsage().rawCharacters, tokenizer.decode(ids, false, false).length);
for (const invalid of [[65], [256, 258], [256, 256], [256, 257, 257], [256], [256, 257, 258],
  [256, 257, 32, 258], [256, 257, 10, 258]]) {
  assert.throws(() => decodeGenerationOutput(tokenizer, contract, [258], invalid),
    error => error.code === 'DOPPLER_GENERATION_OUTPUT_INVALID');
}
assert.equal(decodeGenerationOutput(tokenizer, contract, [258], [256, 257, ...bytes('\n Yes\n'), 258]), '\n Yes\n');
for (const invalid of [{ ...contract, extra: true }, { ...contract, schema: 'unknown' },
  { ...contract, closingToken: contract.openingToken }, { ...contract, openingToken: { id: -1, text: 'x' } }]) {
  assert.throws(() => validateReasoningOutput(invalid));
}
assert.throws(() => createGenerationOutputDecoder(tokenizer, { ...contract, closingToken: { id: 257, text: 'wrong' } }, []), /markers/);
assert.throws(() => createGenerationOutputDecoder(tokenizer, contract, [257]), /stop/);
assert.throws(() => createGenerationOutputDecoder({ decode: tokenizer.decode.bind(tokenizer) }, contract, []), /raw incremental/);

// Cancellation discards incomplete reasoning, retaining the actual abort cause.
let settled = 0;
async function* raw(values) { try { yield* values; } finally { settled++; } }
const abort = new AbortController();
const cancelled = decoder();
const stream = decodeGenerationStream(raw([256, 65]), cancelled,
  { signal: abort.signal, onToken() { abort.abort(new Error('cancelled by caller')); } });
await assert.rejects(stream.next(), /cancelled by caller/);
assert.equal(cancelled.getUsage().state, 'reasoning');
assert.equal(settled, 1);
assert.deepEqual(await Array.fromAsync(decodeGenerationStream(raw(ids), decoder())), ['Y', 'e', 's', ' ', '🌍']);

// Existing Capsule v1/v2 formats expose identical final text and all raw IDs.
for (const version of [1, 2]) {
  const program = { createIncrementalDecoder: decoder, getTokenContract: () => ({ reasoningOutput: contract }),
    decodeTokens() { throw new Error('must use stateful output decoding'); } };
  const adapters = createCapsuleOperationAdapters({ program, generate: () => raw(ids) });
  const iterator = adapters.generate.execute({ schema: `doppler.capsule-operation-request/v${version}`, input: {}, options: {} },
    new AbortController().signal);
  let final;
  while (true) { const item = await iterator.next(); if (item.done) { final = item.value; break; }
    assert(!JSON.stringify(item.value).includes('private'));
  }
  assert.equal(final.text, expected); assert.deepEqual(final.tokenIds, ids);
  const controller = new AbortController();
  const stopped = adapters.generate.execute({ schema: `doppler.capsule-operation-request/v${version}`, input: {}, options: {} }, controller.signal);
  await stopped.next(); controller.abort(new Error('explicit cancellation'));
  await assert.rejects(stopped.next(), /explicit cancellation/);
}

const pipeline = Object.create(PipelineGenerator.prototype);
pipeline._state = { stats: {}, tokenizer, modelConfig: { reasoningOutput: contract, stopTokenIds: [258] } };
pipeline._generateTokensInternal = (prompt, options, mode) => {
  assert.equal(mode, 'token'); return raw(ids);
};
assert.equal((await Array.fromAsync(pipeline.generate('prompt'))).join(''), expected);
assert.equal(pipeline._state.stats.outputUsage.rawTokens, ids.length);
assert.deepEqual(await Array.fromAsync(pipeline.generateTokens('prompt')), ids, 'raw token API retains reasoning IDs');
const retainedUsage = pipeline._state.stats.outputUsage;
pipeline._generateTokensInternal = async function* () { throw new Error('Generation already in progress'); };
await assert.rejects(pipeline.generate('other prompt').next(), /already in progress/);
assert.equal(pipeline._state.stats.outputUsage, retainedUsage, 'Rejected concurrent calls cannot replace active usage');
const handle = createModelHandle({ tokenizer, modelConfig: { reasoningOutput: contract, stopTokenIds: [258] },
  manifest: { inference: { output: { reasoningOutput: contract } } } }, { modelId: 'fixture' });
assert.equal(handle.advanced.decodeTokenIds([256, 65]), '');
assert.equal(handle.advanced.decodeTokenIds(ids), expected);
const handleDecoder = handle.advanced.createIncrementalDecoder();
assert.equal(ids.map(id => handleDecoder.push(id)).join('') + handleDecoder.finish(), expected);
for (const stopReason of ['max-tokens', 'stop-sequence']) {
  const partial = decoder(); partial.push(256); partial.push(65);
  assert.equal(partial.finish({ stopReason }), '');
  assert.equal(partial.getUsage().state, 'truncated');
  assert.equal(partial.getUsage().rawTokens, 2);
}
console.log('generation-output.test: ok (synthetic decoding and control, no model qualification)');
