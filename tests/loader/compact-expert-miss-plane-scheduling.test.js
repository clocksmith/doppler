import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { CompactExpertCache } from '../../src/loader/experts/compact-expert-cache.js';
import { createRoutedExpertCounters } from '../../src/loader/experts/routed-expert-source.js';
import { createVerifiedPieceGather } from '../../src/storage/verified-piece-gather.js';
import { DEFAULT_EXPERT_CACHE_CONFIG } from '../../src/config/schema/runtime/loading.schema.js';

globalThis.GPUBufferUsage ??= { STORAGE: 128, COPY_SRC: 4, COPY_DST: 8 };
const hash = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const encode = value => new TextEncoder().encode(JSON.stringify(value));
function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}

test('small miss sets authenticate six owned planes concurrently and settle failures before publication', { timeout: 5000 }, async () => {
  const directory = await mkdtemp(join(tmpdir(), 'compact-miss-planes-'));
  const filename = join(directory, 'weights.bin');
  const bytes = Uint8Array.from({ length: 96 }, (_, i) => i + 1);
  const manifestBytes = encode({ shards: [{ filename: 'weights.bin', size: bytes.length }] });
  const indexBytes = encode({ schema: 'doppler.verified-pieces/v1', manifestIdentity: hash(manifestBytes), maxPieceBytes: 8,
    files: [{ path: 'weights.bin', size: bytes.length, pieces: Array.from({ length: 12 }, (_, i) => ({
      offset: i * 8, size: 8, identity: hash(bytes.subarray(i * 8, i * 8 + 8)),
    })) }] });
  const plan = { layer: 0, numLayers: 1, numExperts: 4, expertBytes: 24, intermediateSize: 2,
    planes: ['gate', 'up', 'down'].map((role, i) => ({ role, dtype: 'f32', name: role, expertBytes: 8,
      location: { shardIndex: 0, offset: i * 32, size: 32, shape: [4, 2, 1] } })) };
  const policy = { ...structuredClone(DEFAULT_EXPERT_CACHE_CONFIG.qwen35Routed), enabled: true,
    storageMode: 'compact-decode', sourceReadMode: 'expert-lease-batch', expertBatchSize: 1,
    cacheBytes: 112, stagingBytes: 512, readConcurrency: 8, gpuBudgetBytes: 4096, reserveBytes: 16 };
  const state = { calls: [], leases: [], publications: 0, uploads: 0, failure: false,
    entry: null, held: null, heldStarted: null, siblingAborted: null };
  const device = { limits: { maxBufferSize: 4096, maxStorageBufferBindingSize: 4096 }, createBuffer(description) {
    return { ...description, data: new Uint8Array(description.size), destroy() {} };
  }, queue: { async onSubmittedWorkDone() {}, writeBuffer(target, offset, payload) {
    if (target.label.endsWith('_source_map')) state.publications++;
    else {
      state.uploads++;
      if (state.failure) { state.failure = false; throw Error('injected plane upload failure'); }
    }
    target.data.set(new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength), offset);
  } } };
  let worker, cache;
  try {
    await writeFile(filename, bytes);
    worker = await createVerifiedPieceGather({ manifestBytes, indexBytes, indexIdentity: hash(indexBytes),
      files: { 'weights.bin': filename }, maxPartBytes: 24, memoryBudgetBytes: 320, concurrency: 8 });
    const rangeLease = { ...worker.rangeLease, async readMany(ranges, signal) {
      state.calls.push(ranges.map(range => ({ ...range })));
      if (state.entry) await state.entry.promise;
      let lease;
      try { lease = await worker.rangeLease.readMany(ranges, signal); }
      catch (error) {
        if (state.held) await state.heldStarted.promise;
        throw error;
      }
      const record = { lease, releases: 0 };
      state.leases.push(record);
      if (state.held && ranges[0].offset === 40) {
        state.heldStarted.resolve();
        signal.addEventListener('abort', () => state.siblingAborted?.resolve(), { once: true });
        if (signal.aborted) state.siblingAborted?.resolve();
        await state.held.promise;
      } else if (state.held) await state.heldStarted.promise;
      return { bytes: lease.bytes, release() {
        assert.equal(++record.releases, 1, 'each original plane lease releases exactly once');
        lease.release();
      } };
    } };
    const options = { policy, device, compactLayerCount: 1, otherGpuBytes: () => 0, rangeLease,
      readRange: async () => { throw Error('generic reads must not run'); } };
    const fresh = changes => {
      cache = new CompactExpertCache({ ...options, ...changes }, createRoutedExpertCounters());
      return cache;
    };
    const prepare = (ids = [0, 1], signal = new AbortController().signal) => cache.prepare(plan, ids, signal);
    const verify = (weights, ids = [0, 1]) => {
      const slots = new Uint32Array(weights.sourceExpertToSlot.data.buffer);
      for (const expert of ids) for (const [i, role] of ['gate', 'up', 'down'].entries()) {
        assert.deepEqual(weights[role].buffer.data.slice(slots[expert] * 8, slots[expert] * 8 + 8),
          bytes.slice(i * 32 + expert * 8, i * 32 + expert * 8 + 8));
      }
      assert(state.leases.every(record => record.releases === 1 && record.lease.bytes.byteLength === 0));
    };
    fresh(); state.entry = deferred();
    const pending = prepare();
    assert.deepEqual(state.calls.map(ranges => ranges.length), [1, 1, 1, 1, 1, 1]);
    assert.equal(state.uploads, 0, 'no expert uploads before its three planes authenticate');
    assert.equal(state.publications, 0, 'source map cannot publish ahead of sibling completion');
    state.entry.resolve(); state.entry = null;
    verify(await pending);
    const previousReads = state.calls.length;
    verify(await prepare());
    assert.equal(state.calls.length, previousReads, 'repeat resident requests issue no acquisitions');
    await cache.destroy();

    // A corrupted expert aborts the other task, but the held original lease
    // must settle and release before prepare rejects or publishes readiness.
    fresh(); state.held = deferred(); state.heldStarted = deferred(); state.siblingAborted = deferred();
    const corrupt = bytes.slice(); corrupt[64] ^= 128;
    await writeFile(filename, corrupt);
    const beforeCorrupt = state.publications;
    const corruptRequest = prepare();
    const corruptRejection = assert.rejects(corruptRequest, /integrity/);
    let settled = false;
    corruptRequest.then(() => { settled = true; }, () => { settled = true; });
    await state.heldStarted.promise; await state.siblingAborted.promise;
    assert.equal(settled, false, 'corruption waits for every admitted sibling to settle');
    assert.equal(state.publications, beforeCorrupt);
    state.held.resolve(); state.held = null;
    await corruptRejection;
    await writeFile(filename, bytes);
    verify(await prepare()); await cache.destroy();

    for (const failure of ['cancel', 'upload']) {
      fresh(); state.held = deferred(); state.heldStarted = deferred(); state.siblingAborted = deferred();
      const controller = new AbortController(), before = state.publications;
      state.failure = failure === 'upload';
      const failing = prepare([0, 1], controller.signal);
      const rejection = assert.rejects(failing, failure === 'cancel' ? /cancel small miss set/ : /plane upload failure/);
      let done = false; failing.then(() => { done = true; }, () => { done = true; });
      await state.heldStarted.promise;
      if (failure === 'cancel') controller.abort(Error('cancel small miss set'));
      await state.siblingAborted.promise;
      assert.equal(done, false, 'failure retains sibling ownership until actual settlement');
      assert.equal(state.publications, before);
      state.held.resolve(); state.held = null;
      await rejection;
      verify(await prepare()); await cache.destroy();
    }

    // Insufficient policy or port lanes preserve whole-expert fallback, as do
    // larger miss sets and mapped-cache hooks which require their old owner.
    for (const changes of [
      { policy: { ...policy, readConcurrency: 4 } },
      { rangeLease: { ...rangeLease, concurrency: 4 } },
      { copyCachedExpert: () => false },
      { rememberExpert: () => {} },
    ]) {
      fresh(changes); state.calls.length = 0;
      verify(await prepare());
      assert.deepEqual(state.calls.map(ranges => ranges.length), [3, 3]);
      await cache.destroy();
    }
    fresh(); state.calls.length = 0;
    verify(await prepare([0, 1, 2]), [0, 1, 2]);
    assert.deepEqual(state.calls.map(ranges => ranges.length), [3, 3, 3]);
  } finally {
    state.entry?.resolve(); state.held?.resolve();
    await cache?.destroy(); await worker?.close(); await rm(directory, { recursive: true, force: true });
  }
});
