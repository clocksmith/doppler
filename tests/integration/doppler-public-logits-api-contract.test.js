import assert from 'node:assert/strict';
import { createModelHandle } from '../../src/client/model-host/model-session.js';
import { load as nodeLoad } from '../../src/client/doppler-api.js';
import { load as browserLoad } from '../../src/client/doppler-api.browser.js';

// Both public entrypoints delegate to the shared model-session owner. Exercise
// its methods and lifecycle, rather than requiring their bodies in each facade.
assert.equal(typeof nodeLoad, 'function');
assert.equal(typeof browserLoad, 'function');
const calls = [];
const prefill = { tokens: [7], seqLen: 1, logits: new Float32Array([1, 2]) };
const decoded = { logits: new Float32Array([3, 4]), vocabSize: 2, rawVocabSize: 2 };
const failure = new Error('physical execution failed');
let fail = false;
const pipeline = {
  async prefillWithLogits(prompt, options) {
    calls.push(['prefill', prompt, options]);
    if (fail) throw failure;
    return prefill;
  },
  async decodeStepLogits(ids, options) {
    calls.push(['decode', ids, options]);
    if (fail) throw failure;
    return decoded;
  },
  async unload() {},
};
const handle = createModelHandle(pipeline, { modelId: 'logits-contract' });
const prompt = [{ role: 'user', content: 'test' }];
const options = { useChatTemplate: false };
assert.equal(await handle.advanced.prefillWithLogits(prompt, options), prefill);
assert.equal(await handle.advanced.decodeStepLogits([7], options), decoded);
assert.deepEqual(calls, [['prefill', prompt, options], ['decode', [7], options]]);
assert.throws(() => handle.advanced.prefillWithLogits('x', { stopTokens: [1] }), /stopSequences/);
assert.equal(calls.length, 2, 'invalid options do not execute');
fail = true;
await assert.rejects(handle.advanced.prefillWithLogits('x'), error => error === failure);
await assert.rejects(handle.advanced.decodeStepLogits([7]), error => error === failure);
fail = false;
assert.equal(await handle.advanced.prefillWithLogits('retry'), prefill, 'failed calls release their operation lease');
await handle.unload();
await assert.rejects(handle.advanced.decodeStepLogits([7]), /closing or closed/);
console.log('doppler-public-logits-api-contract: shared public handle behavior passed (stubbed execution)');
