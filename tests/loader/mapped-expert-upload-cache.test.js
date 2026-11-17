import assert from 'node:assert/strict';
import test from 'node:test';
import { MappedExpertUploadCache } from '../../src/loader/experts/mapped-expert-upload-cache.js';

globalThis.GPUBufferUsage = { MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8 };
const turn = () => new Promise(resolve => setImmediate(resolve));
function fixture(overrides = {}) {
  const state = { buffers: [], failAllocate: false, failDispose: false, failUnmap: false, failFence: false,
    failSubmit: false, fence: Promise.resolve(), submissions: 0, liveBytes: 0 };
  let lose;
  const device = { limits: { maxBufferSize: 1024 * 1024 }, lost: new Promise(resolve => { lose = resolve; }),
    createBuffer(descriptor) {
      if (state.failAllocate) throw new Error('allocation failed');
      const buffer = { size: descriptor.size, usage: descriptor.usage, mapped: descriptor.mappedAtCreation === true,
        destroyed: false, storage: new Uint8Array(descriptor.size),
        getMappedRange(offset, bytes) {
          assert(this.mapped && !this.destroyed); assert.equal(offset, 0); assert.equal(bytes, this.size);
          return this.storage.buffer;
        },
        unmap() {
          if (state.failUnmap) throw new Error('unmap failed');
          assert(this.mapped && !this.destroyed);
          this.storage = new Uint8Array(structuredClone(this.storage.buffer, { transfer: [this.storage.buffer] }));
          this.mapped = false;
        },
        destroy() {
          if (state.failDispose) throw new Error('dispose failed');
          if (this.destroyed) return;
          this.destroyed = true; this.mapped = false; state.liveBytes -= this.size;
          this.storage = new Uint8Array(structuredClone(this.storage.buffer, { transfer: [this.storage.buffer] }));
        } };
      state.buffers.push(buffer); state.liveBytes += buffer.size;
      return buffer;
    },
    createCommandEncoder() {
      const commands = [];
      return { copyBufferToBuffer(source, from, target, to, bytes) {
        commands.push(() => { assert(!source.destroyed && !source.mapped); target.storage.set(source.storage.subarray(from, from + bytes), to); });
      }, finish: () => commands };
    },
    queue: { submit(commands) {
      state.submissions++;
      for (const list of commands) for (const op of list) op();
      if (state.failSubmit) throw new Error('submission uncertain');
    }, onSubmittedWorkDone: () => state.failFence ? Promise.reject(new Error('fence failed')) : state.fence },
  };
  const cache = new MappedExpertUploadCache({ device, maxBytes: 24, maxMappedBytes: 16, maxEntries: 3, ...overrides });
  const put = (key, value = 1, options = {}) => cache.admit(key, 8, destination => { destination.fill(value); return 8; }, options);
  const destination = bytes => device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST });
  const copies = target => [{ sourceOffset: 0, destination: target, destinationOffset: 0, bytes: 8 }];
  return { device, cache, state, put, destination, copies, lose };
}

test('authenticated caller payload publishes only unmapped, with pinned entries and deterministic idle LRU', async () => {
  const { cache, state, put, destination, copies } = fixture();
  let borrowed;
  await cache.admit('model:index:parked', 8, view => { borrowed = view; view.fill(17); return 8; }, { pinned: true });
  assert.equal(borrowed.byteLength, 0, 'unmap invalidates the borrowed write mapping before publication');
  await put('A', 21); await put('B', 33);
  const target = destination(8);
  await cache.copyInto('A', copies(target)).settled;
  assert.deepEqual([...target.storage], new Array(8).fill(21));
  await put('C', 55);
  assert.equal(cache.copyInto('B', copies(target)), null, 'least recently used idle entry is evicted');
  await cache.copyInto('model:index:parked', copies(target)).settled;
  assert.deepEqual([...target.storage], new Array(8).fill(17));
  assert.equal(cache.getStats().pinnedEntries, 1);
  assert.equal(cache.getStats().evictions, 1);
  assert.equal(cache.getStats().allocatedBytes, 24);
  assert.equal(cache.getStats().mappedBytes, 0);
  assert.ok(state.buffers.filter(buffer => buffer.usage & GPUBufferUsage.MAP_WRITE).every(buffer => buffer.usage === 6));
  await cache.close(); target.destroy();
  assert.equal(state.liveBytes, 0);
});

test('in-flight submitted sources cannot be evicted; copy descriptors are snapshotted', async () => {
  const { cache, state, put, destination, copies } = fixture({ maxBytes: 8, maxMappedBytes: 8, maxEntries: 1 });
  await put('A', 42);
  let complete; state.fence = new Promise(resolve => { complete = resolve; });
  const target = destination(8), ranges = copies(target);
  const ticket = cache.copyInto('A', ranges);
  ranges[0].bytes = 4; ranges.length = 0;
  assert.equal(ticket.bytes, 8);
  assert.equal(cache.getStats().pendingCopies, 1);
  assert.equal(await put('B'), false);
  assert.equal(cache.getStats().evictions, 0);
  complete(); await ticket.settled;
  assert.equal(await put('B'), true);
  let finishClose; state.fence = new Promise(resolve => { finishClose = resolve; });
  const next = cache.copyInto('B', copies(target));
  let closed = false; const closing = cache.close().then(() => { closed = true; });
  await turn();
  assert.equal(closed, false, 'close waits an actual submitted-copy completion');
  assert.equal(cache.getStats().allocatedBytes, 8);
  finishClose(); await next.settled; await closing; target.destroy();
  assert.equal(state.liveBytes, 0);
});

test('mapped writers obey their separate aggregate allowance and close waits a cancellation-ignoring writer', async () => {
  const { cache, state, put } = fixture({ maxMappedBytes: 8 });
  let finish;
  const writing = cache.admit('pending', 8, async view => {
    await new Promise(resolve => { finish = resolve; }); view.fill(9); return 8;
  });
  await turn();
  assert.equal(cache.getStats().mappedBytes, 8);
  assert.equal(await put('other'), false);
  assert.equal(cache.getStats().allocatedBytes, 8);
  const closing = cache.close();
  assert.equal(cache.getStats().mappedBytes, 8, 'close cannot discard a writer mapping before the callback settles');
  assert.equal(state.buffers[0].destroyed, false);
  assert.throws(() => put('late'), /closing/);
  finish();
  await assert.rejects(writing, /closed/); await closing;
  assert.equal(cache.getStats().allocatedBytes, 0);
  assert.equal(cache.getStats().mappedBytes, 0);
  assert.equal(state.liveBytes, 0);
});

test('partial fill, external cancellation and allocation/unmap failure never publish a usable source', async () => {
  const { cache, state, put, destination, copies } = fixture();
  await assert.rejects(cache.admit('partial', 8, view => { view[0] = 1; return 4; }), /complete.*extent/);
  const abort = new AbortController();
  await assert.rejects(cache.admit('cancelled', 8, view => { view.fill(2); abort.abort(new Error('source cancelled')); return 8; }, { signal: abort.signal }), /source cancelled/);
  state.failAllocate = true; await assert.rejects(put('allocation'), /allocation failed/); state.failAllocate = false;
  state.failUnmap = true; await assert.rejects(put('unmap'), /unmap failed/); state.failUnmap = false;
  assert.equal(cache.getStats().entries, 0);
  assert.equal(cache.getStats().allocatedBytes, 0);
  const target = destination(8);
  assert.equal(cache.copyInto('partial', copies(target)), null);
  await put('recovered', 7); await cache.copyInto('recovered', copies(target)).settled;
  assert.deepEqual([...target.storage], new Array(8).fill(7));
  await cache.close(); target.destroy(); assert.equal(state.liveBytes, 0);
});

test('failed completion and disposal keep allocations charged; close remains retryable', async () => {
  const { cache, state, put, destination, copies } = fixture({ maxBytes: 8, maxMappedBytes: 8, maxEntries: 1 });
  await put('A'); const target = destination(8);
  state.failFence = true;
  await assert.rejects(cache.copyInto('A', copies(target)).settled, /fence failed/);
  await assert.rejects(cache.close(), /retirement failed/);
  assert.equal(cache.getStats().allocatedBytes, 8);
  assert.equal(state.buffers[0].destroyed, false);
  state.failFence = false; state.failDispose = true;
  await assert.rejects(cache.close(), /retirement failed/);
  assert.equal(cache.getStats().allocatedBytes, 8);
  state.failDispose = false; await cache.close(); await cache.close();
  target.destroy(); assert.equal(state.liveBytes, 0);
});

test('failed partial-writer disposal retains both persistent and mapped charges until retry', async () => {
  const { cache, state } = fixture();
  state.failDispose = true;
  await assert.rejects(cache.admit('unpublished', 8, view => { view[0] = 77; throw new Error('partial source failure'); }), /admission and retirement failed/);
  assert.equal(cache.getStats().entries, 0);
  assert.equal(cache.getStats().unpublishedBytes, 8);
  assert.equal(cache.getStats().allocatedBytes, 8);
  assert.equal(cache.getStats().mappedBytes, 8);
  await assert.rejects(cache.close(), /retirement failed/);
  assert.equal(cache.getStats().mappedBytes, 8);
  state.failDispose = false;
  await cache.close();
  assert.equal(cache.getStats().mappedBytes, 0);
  assert.equal(cache.getStats().allocatedBytes, 0);
  assert.equal(state.liveBytes, 0);
});

test('uncertain submission still owns its source until a real fence; known device loss can retire it', async () => {
  const { cache, state, put, destination, copies, lose } = fixture();
  await put('A'); const target = destination(8);
  state.failSubmit = true; state.failFence = true;
  assert.throws(() => cache.copyInto('A', copies(target)), /submission uncertain/);
  await turn();
  assert.equal(cache.getStats().pendingCopies, 1);
  lose({ reason: 'destroyed' }); await turn();
  await cache.close(); target.destroy(); assert.equal(state.liveBytes, 0);
});

test('a reentrant close cannot bypass writer ownership; an unsuccessful admission does not evict useful entries', async () => {
  const { cache, state, put } = fixture({ maxBytes: 16, maxMappedBytes: 16, maxEntries: 2 });
  await put('pinned', 1, { pinned: true }); await put('useful', 2);
  assert.equal(await cache.admit('too-big', 16, () => 16), false);
  assert.equal(cache.getStats().entries, 2);
  assert.equal(cache.getStats().evictions, 0);
  let closing;
  const writing = cache.admit('reentrant', 8, () => { closing = cache.close(); return 8; });
  await assert.rejects(writing, /closed/); await closing;
  assert.equal(state.liveBytes, 0);
});

test('constructor bounds, exact key extent and copy ranges fail closed', async () => {
  assert.throws(() => fixture({ maxMappedBytes: 28 }), /explicit aligned/);
  const { cache, put, destination, copies, state } = fixture();
  await put('A');
  await assert.rejects(cache.admit('A', 12, () => 12), /changed its byte extent/);
  const target = destination(8), valid = copies(target);
  assert.throws(() => cache.copyInto('A', [{ ...valid[0], bytes: 12 }]), /exceeds/);
  assert.throws(() => cache.copyInto('A', [{ ...valid[0], sourceOffset: 1 }]), /exceeds/);
  assert.throws(() => cache.copyInto('A', new Array(7).fill(valid[0])), /one to six/);
  assert.throws(() => cache.copyInto('A', [{ ...valid[0], destination: state.buffers[0] }]), /exceeds/);
  await cache.close(); target.destroy(); assert.equal(state.liveBytes, 0);
});
