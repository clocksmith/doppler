import assert from 'node:assert/strict';
import { createCapsuleSessionExecution } from '../../src/client/runtime/capsule-session-execution.js';

const deferred = () => { let resolve; const promise = new Promise(done => { resolve = done; }); return { promise, resolve }; };
const scope = createCapsuleSessionExecution();
const work = deferred(), entered = deferred();
let disposed = false;
const running = scope.run(async signal => { entered.resolve(signal); await work.promise; return 'late'; });
await entered.promise;
assert.throws(() => scope.run(() => 'overlap'), /already active/);
const cancelled = assert.rejects(running, /session is closed/);
const close = scope.close(() => { disposed = true; });
assert.equal((await entered.promise).aborted, true);
await Promise.resolve();
assert.equal(disposed, false, 'abort cannot dispose resources still used by physical work');
assert.equal(scope.close(() => assert.fail('duplicate disposal')), close);
work.resolve();
await cancelled;
await close;
assert.equal(disposed, true);
assert.throws(() => scope.run(() => 'after close'), /session is closed/);

const streaming = createCapsuleSessionExecution();
const order = [];
const iterator = streaming.stream(async function* () {
  try { yield 'partial'; } finally { order.push('drained'); throw new Error('adapter unload failed'); }
});
await iterator.next();
await assert.rejects(streaming.close(() => { order.push('disposed'); }), /adapter unload failed/);
assert.deepEqual(order, ['drained', 'disposed']);

const failure = createCapsuleSessionExecution();
const caller = new AbortController();
assert.throws(() => failure.run(() => {}, {}), /AbortSignal/);
assert.equal(failure.run(() => 'recovered'), 'recovered');
const rejected = failure.run(async signal => {
  caller.abort(new Error('caller cancelled'));
  assert.equal(signal.reason, caller.signal.reason);
}, caller.signal);
await assert.rejects(rejected, /caller cancelled/);
assert.equal(failure.run(() => 'recovered'), 'recovered');
await failure.close(() => {});
console.log('capsule-session-execution: exclusion, cancellation, draining and cleanup failure passed');
