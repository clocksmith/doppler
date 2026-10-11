import assert from 'node:assert/strict';
import { qualifyEmbeddingCancellation } from '../../tools/lib/embedding-qualification-lifecycle.js';

const clean = () => ({ active: { bytes: 16, count: 1 }, deferredCleanup: { bytes: 0, count: 0 } });
async function check(run, resourceSequence = [clean(), clean()]) {
  let settled = false, submitted = false;
  const queue = { submit() { submitted = true; }, async onSubmittedWorkDone() { settled = true; } };
  const original = queue.submit;
  let reads = 0;
  try { return await qualifyEmbeddingCancellation({ queue, run: signal => run(queue, signal),
    resources: () => resourceSequence[Math.min(reads++, resourceSequence.length - 1)] }); }
  finally { assert.equal(queue.submit, original); assert.equal(settled, true); }
}
const passed = await check(async (queue, signal) => {
  assert.equal(signal.aborted, false); queue.submit([]); assert.equal(signal.aborted, true); signal.throwIfAborted();
});
assert.equal(passed.passed, true); assert.equal(passed.submissions, 1);
await assert.rejects(check(async queue => { queue.submit([]); }), /did not propagate/);
await assert.rejects(check(async () => {}), /did not follow/);
await assert.rejects(check(async queue => { queue.submit([]); throw new Error('unrelated failure'); }), /unrelated failure/);
await assert.rejects(check(async (queue, signal) => { queue.submit([]); signal.throwIfAborted(); },
  [clean(), { ...clean(), active: { bytes: 32, count: 2 } }]), /retained active/);
await assert.rejects(check(async (queue, signal) => { queue.submit([]); signal.throwIfAborted(); },
  [clean(), { ...clean(), deferredCleanup: { bytes: 4, count: 1 } }]), /retained active/);
console.log('embedding-qualification-lifecycle.test: passed');
