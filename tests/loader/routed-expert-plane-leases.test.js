import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createRoutedExpertCounters, readRoutedExpertPlaneLeases } from '../../src/loader/experts/routed-expert-source.js';

function plan() {
  return { numExperts: 2, expertBytes: 12, planes: [0, 1, 2].map(shardIndex => ({
    expertBytes: 4, location: { shardIndex, offset: 0, size: 8 },
  })) };
}
function options(readMany, changes = {}) {
  return { policy: { sourceReadMode: 'expert-lease-batch', expertBatchSize: 1 },
    rangeLease: { concurrency: 3, maxPartBytes: 12, maxRanges: 3, readMany, ...changes } };
}
function deferred() {
  let resolve, reject;
  const promise = new Promise((accept, fail) => { resolve = accept; reject = fail; });
  return { promise, resolve, reject };
}
function leases(released, lengths = [4, 4, 4]) {
  return lengths.map((length, index) => ({ bytes: new Uint8Array(length).fill(index),
    release() { released.push(index); } }));
}
const tick = () => new Promise(resolve => setImmediate(resolve));
const signal = () => new AbortController().signal;

test('plane leases start together, preserve caller order without copies and account wall wait once', { timeout: 2000 }, async () => {
  const waits = [deferred(), deferred(), deferred()], calls = [], released = [];
  const owned = leases(released), stats = createRoutedExpertCounters();
  const layer = { sourceReadBytes: 0, sourceReadWaitMs: 0 };
  let settled = false;
  const port = options(function(parts) {
    assert.equal(this, port.rangeLease, 'Validated range method preserves its owning port');
    calls.push(parts); return waits[calls.length - 1].promise;
  });
  const pending = readRoutedExpertPlaneLeases(port, stats, plan(), 1, signal(), layer)
    .then(value => { settled = true; return value; });
  assert.deepEqual(calls, [0, 1, 2].map(shard => [{ shard, offset: 4, length: 4 }]));
  waits[2].resolve(owned[2]); waits[0].resolve(owned[0]); await tick();
  assert.equal(settled, false); assert.deepEqual(released, []); assert.equal(stats.sourceReads, 0);
  waits[1].resolve(owned[1]); const result = await pending;
  assert(result.every((lease, index) => lease === owned[index]), 'Return exact lease objects without concatenating plane payloads');
  assert.equal(stats.sourceReads, 3); assert.equal(stats.sourceReadBytes, 12);
  assert.equal(layer.sourceReadBytes, 12); assert.equal(layer.sourceReadWaitMs, stats.sourceReadWaitMs);
  assert.equal(stats.sourceReadIntoBytes, 0); assert.equal(stats.stagingAllocations, 0);
  assert.deepEqual(released, [], 'Successful leases belong to the caller through uploads');
  result.forEach(lease => lease.release()); assert.deepEqual(released, [0, 1, 2]);
});

test('synchronous admission failure still starts every sibling and waits before releasing fulfilled leases', { timeout: 2000 }, async () => {
  const last = deferred(), released = [], owned = leases(released), stats = createRoutedExpertCounters();
  let calls = 0, settled = false;
  const failure = new Error('synchronous admission failure');
  const pending = readRoutedExpertPlaneLeases(options(() => {
    const index = calls++;
    if (!index) throw failure;
    return index === 1 ? owned[index] : last.promise;
  }), stats, plan(), 1, signal(), null);
  const rejected = assert.rejects(pending, error => error === failure).then(() => { settled = true; });
  assert.equal(calls, 3); await tick();
  assert.equal(settled, false); assert.deepEqual(released, []);
  last.resolve(owned[2]); await rejected;
  assert.deepEqual(released, [1, 2]); assert.equal(stats.sourceReads, 0); assert.equal(stats.sourceReadBytes, 0);
});

test('failure cleanup attempts every fulfilled release and preserves acquisition and release failures', { timeout: 2000 }, async () => {
  const released = [], owned = leases(released), acquisition = new Error('piece authentication failure');
  const cleanup = new Error('first lease retirement failed');
  owned[0].release = () => { released.push(0); throw cleanup; };
  const stats = createRoutedExpertCounters(); let calls = 0;
  await assert.rejects(readRoutedExpertPlaneLeases(options(() => {
    const index = calls++; return index === 1 ? Promise.reject(acquisition) : owned[index];
  }), stats, plan(), 1, signal(), null), error => {
    assert(error instanceof AggregateError); assert.deepEqual(error.errors, [acquisition, cleanup]); return true;
  });
  assert.deepEqual(released, [0, 2]); assert.equal(calls, 3); assert.equal(stats.sourceReads, 0);
});

test('a short authenticated plane rejects the entire result and releases every fulfilled lease', { timeout: 2000 }, async () => {
  const released = [], owned = leases(released, [4, 3, 4]), stats = createRoutedExpertCounters(); let calls = 0;
  await assert.rejects(readRoutedExpertPlaneLeases(options(() => owned[calls++]), stats,
    plan(), 1, signal(), null), /range length mismatch/);
  assert.deepEqual(released, [0, 1, 2]); assert.equal(stats.sourceReadBytes, 0);
});

test('abort retains all owned operations until the final sibling settles and prevents publication', { timeout: 2000 }, async () => {
  const last = deferred(), controller = new AbortController(), released = [], owned = leases(released);
  const stats = createRoutedExpertCounters(), cancelled = new Error('cancel all expert planes');
  let calls = 0, settled = false;
  const pending = readRoutedExpertPlaneLeases(options((parts, activeSignal) => {
    assert.equal(activeSignal, controller.signal);
    const index = calls++; return index === 2 ? last.promise : owned[index];
  }), stats, plan(), 1, controller.signal, null);
  const rejected = assert.rejects(pending, error => error === cancelled).then(() => { settled = true; });
  await tick(); controller.abort(cancelled); await tick();
  assert.equal(settled, false); assert.deepEqual(released, []);
  last.resolve(owned[2]); await rejected;
  assert.deepEqual(released, [0, 1, 2]); assert.equal(stats.sourceReads, 0);
  calls = 0;
  await assert.rejects(readRoutedExpertPlaneLeases(options(() => { calls++; }), stats, plan(), 1,
    controller.signal, null), error => error === cancelled);
  assert.equal(calls, 0, 'Pre-aborted calls do not acquire any source lease');
});

test('lane admission, unchanged plane geometry and physical bounds reject before any acquisition', { timeout: 2000 }, async () => {
  let calls = 0;
  const make = changes => options(() => { calls++; throw new Error('Unexpected acquisition'); }, changes);
  const cases = [
    { options: make({ concurrency: 2 }) }, { options: make({ concurrency: 1.5 }) },
    { options: make({ concurrency: undefined }) }, { options: make({ maxPartBytes: 11 }) },
    { options: make({ maxRanges: 2 }) }, { options: make({ readMany: undefined }) },
    { change: value => { value.planes[0].location.sourceTransform = {}; } },
    { change: value => { value.planes[0].location.spans = [{ shardIndex: 0, offset: 0, size: 4 }, { shardIndex: 1, offset: 0, size: 4 }]; } },
    { change: value => { value.planes[0].location.size = 4; } },
    { change: value => { value.planes[0].location.shardIndex = -1; } },
    { change: value => { value.planes[0].location.offset = 1; } },
    { change: value => { value.planes[0].expertBytes = 3; } },
    { expert: -1 }, { expert: 2 }, { expert: 1.5 },
  ];
  for (const value of cases) {
    const selected = plan(); value.change?.(selected);
    await assert.rejects(readRoutedExpertPlaneLeases(value.options ?? make(), createRoutedExpertCounters(),
      selected, value.expert ?? 1, signal(), null));
  }
  assert.equal(calls, 0);
});
