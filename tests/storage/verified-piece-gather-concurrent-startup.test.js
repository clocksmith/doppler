import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { createRequire, syncBuiltinESMExports } from 'node:module';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createVerifiedPieceGather } from '../../src/storage/verified-piece-gather.js';

const threads = createRequire(import.meta.url)('node:worker_threads');
const OriginalWorker = threads.Worker;
const hash = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const encode = value => new TextEncoder().encode(JSON.stringify(value));
function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}

test('concurrent startup joins all budgeted admissions and retires every owner before rejection', { timeout: 5000 }, async () => {
  const directory = await mkdtemp(join(tmpdir(), 'gather-concurrent-startup-'));
  const bytes = Uint8Array.of(1, 2, 3, 4);
  const manifestBytes = encode({ shards: [{ filename: 'weights.bin', size: 4 }] });
  const indexBytes = encode({ schema: 'doppler.verified-pieces/v1', manifestIdentity: hash(manifestBytes), maxPieceBytes: 4,
    files: [{ path: 'weights.bin', size: 4, pieces: [{ offset: 0, size: 4, identity: hash(bytes) }] }] });
  const options = { manifestBytes, indexBytes, indexIdentity: hash(indexBytes),
    files: { 'weights.bin': join(directory, 'weights.bin') }, maxPartBytes: 4, memoryBudgetBytes: 48, concurrency: 4 };
  const allWorkers = [];
  let active, reader;
  const next = (count = 4) => {
    active = { count, workers: [], ready: new Map(), barrier: deferred(), requests: [], cleanupFailure: -1 };
    return active;
  };
  const release = (state, index) => {
    const worker = state.workers[index], message = state.ready.get(worker);
    assert(message, 'the real worker reached authenticated startup before release');
    state.ready.delete(worker);
    OriginalWorker.prototype.emit.call(worker, 'message', message);
  };
  const releaseAll = state => {
    for (const [worker, message] of state.ready) {
      state.ready.delete(worker);
      OriginalWorker.prototype.emit.call(worker, 'message', message);
    }
  };
  try {
    await writeFile(join(directory, 'weights.bin'), bytes);
    threads.Worker = class extends OriginalWorker {
      constructor(...args) {
        super(...args);
        this.startupState = active;
        this.startupIndex = active.workers.length;
        active.workers.push(this); allWorkers.push(this);
      }
      emit(event, message, ...args) {
        if (event === 'message' && message?.ready === true) {
          const state = this.startupState;
          state.ready.set(this, message);
          if (state.ready.size === state.count) state.barrier.resolve();
          return true;
        }
        return super.emit(event, message, ...args);
      }
      postMessage(message, ...args) {
        if (message.ranges) this.startupState.requests.push(this.startupIndex);
        return super.postMessage(message, ...args);
      }
      async terminate() {
        const code = await super.terminate();
        if (this.startupState.cleanupFailure === this.startupIndex) {
          this.startupState.cleanupFailure = -1;
          throw Error('injected late startup cleanup failure');
        }
        return code;
      }
    };
    syncBuiltinESMExports();
    let state = next();
    let published = false;
    const opening = createVerifiedPieceGather(options).then(value => { published = true; return value; });
    await state.barrier.promise;
    assert.equal(state.workers.length, 4, 'all admitted owners start without waiting for an earlier ready message');
    assert.equal(published, false);
    release(state, 3); release(state, 1); release(state, 2);
    await Promise.resolve(); assert.equal(published, false, 'publication waits for the final admission');
    release(state, 0); reader = await opening;
    const leases = await Promise.all(Array.from({ length: 4 }, () => reader.readLease('weights.bin', 0, 4)));
    assert.deepEqual(state.requests, [0, 1, 2, 3], 'lane ownership stays in declared order despite reverse readiness');
    for (const lease of leases) { assert.deepEqual([...lease.bytes], [...bytes]); lease.release(); }
    await reader.close(); reader = null;
    assert(state.workers.every(worker => worker.threadId === -1));

    state = next();
    const controller = new AbortController();
    const cancelled = createVerifiedPieceGather({ ...options, startupSignal: controller.signal });
    const cancellation = assert.rejects(cancelled, /cancel concurrent startup/);
    let settled = false; cancelled.then(() => { settled = true; }, () => { settled = true; });
    await state.barrier.promise;
    release(state, 0);
    controller.abort(Error('cancel concurrent startup'));
    await Promise.resolve(); assert.equal(settled, false, 'abort joins every started admission');
    release(state, 2); release(state, 1); release(state, 3);
    await cancellation;
    assert(state.workers.every(worker => worker.threadId === -1), 'late readiness after abort is still owned and retired');

    state = next(); state.cleanupFailure = 0;
    const failed = createVerifiedPieceGather(options);
    const rejection = assert.rejects(failed, error => {
      assert(error instanceof AggregateError);
      assert(error.errors.some(item => /injected startup authentication failure/.test(String(item))));
      assert(error.errors.some(item => /injected late startup cleanup failure/.test(String(item))));
      return true;
    });
    let failedSettled = false; failed.then(() => { failedSettled = true; }, () => { failedSettled = true; });
    await state.barrier.promise;
    OriginalWorker.prototype.emit.call(state.workers[1], 'error', Error('injected startup authentication failure'));
    await Promise.resolve(); assert.equal(failedSettled, false, 'one failed admission does not discard pending owners');
    state.ready.delete(state.workers[1]);
    release(state, 3); release(state, 0); release(state, 2);
    await rejection;
    assert(state.workers.every(worker => worker.threadId === -1), 'cleanup failure cannot skip another ready owner');

    const beforeInvalid = allWorkers.length;
    await assert.rejects(createVerifiedPieceGather({ ...options,
      startupSignal: AbortSignal.abort(Error('preabort startup')) }), /preabort startup/);
    await assert.rejects(createVerifiedPieceGather({ ...options, memoryBudgetBytes: 47 }), /budget/);
    await assert.rejects(createVerifiedPieceGather({ ...options, maxPartBytes: -1 }), /budget/);
    assert.equal(allWorkers.length, beforeInvalid, 'preabort and aggregate invalid budget start no worker');

    state = next(1);
    const single = createVerifiedPieceGather({ ...options, concurrency: undefined, memoryBudgetBytes: 12 });
    await state.barrier.promise; assert.equal(state.workers.length, 1);
    release(state, 0); reader = await single;
    assert.equal(reader.rangeLease.concurrency, 1, 'omission preserves the single-lane contract');
    await reader.close(); reader = null;
  } finally {
    if (active) releaseAll(active);
    threads.Worker = OriginalWorker; syncBuiltinESMExports();
    await reader?.close();
    await Promise.allSettled(allWorkers.map(worker => OriginalWorker.prototype.terminate.call(worker)));
    await rm(directory, { recursive: true, force: true });
  }
});
