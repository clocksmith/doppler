import assert from 'node:assert/strict';
import { summarizeLogits, qualifySubmittedGeneration } from '../../tools/w8a8-qualification-scenario.js';
const result = summarizeLogits(new Float32Array([1000, 999, 998]), 1);
assert.equal(result.top1, 0);
assert.equal(result.targetLogit, 999);
assert.equal(result.vocabulary, 3);
assert(Math.abs(result.nll - (1 + Math.log(1 + Math.exp(-1) + Math.exp(-2)))) < 1e-12);
assert.throws(() => summarizeLogits(new Float32Array([1, NaN]), 0), /Nonfinite/);
assert.throws(() => summarizeLogits(new Float32Array([1, Infinity]), 0), /Nonfinite/);
assert.throws(() => summarizeLogits(new Float32Array([1]), 1), /full vocabulary/);
assert.throws(() => summarizeLogits([1, 2], 0), /full vocabulary/);
console.log('W8A8 model observation checks passed.');

const resources = () => ({ active: { bytes: 0, count: 0 }, deferredCleanup: { bytes: 0, count: 0 } });
let submitted = false, settled = false;
const queue = { submit() { submitted = true; }, async onSubmittedWorkDone() { settled = true; } };
const submit = queue.submit;
const cancellation = await qualifySubmittedGeneration({ queue, resources, stats: () => ({ stopReason: 'aborted' }),
  run: async signal => { assert.equal(signal.aborted, false); queue.submit([]); assert(submitted && signal.aborted); throw signal.reason; } });
assert(cancellation.passed && settled);
assert.equal(queue.submit, submit);
await assert.rejects(qualifySubmittedGeneration({ queue, resources, stats: () => ({ stopReason: 'stop-token' }),
  run: async () => { queue.submit([]); } }), /did not report cancellation/);
assert.equal(queue.submit, submit);
await assert.rejects(qualifySubmittedGeneration({ queue, resources, stats: () => ({ stopReason: 'aborted' }),
  run: async () => { throw new Error('unrelated failure'); } }), /actual GPU submission/);
assert.equal(queue.submit, submit);
console.log('Submitted generation cancellation controls passed.');

let reads = 0;
await assert.rejects(qualifySubmittedGeneration({ queue,
  resources: () => ({ active: { bytes: reads++ ? 4 : 0, count: reads > 1 ? 1 : 0 }, deferredCleanup: { bytes: 0, count: 0 } }),
  stats: () => ({ stopReason: 'aborted' }), run: async signal => { queue.submit([]); throw signal.reason; } }),
error => error.message.includes('retained active') && error.evidence.before.active.bytes === 0 && error.evidence.after.active.bytes === 4);
assert.equal(queue.submit, submit);
