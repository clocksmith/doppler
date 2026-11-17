import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createHash } from 'node:crypto';
import { createRequire, syncBuiltinESMExports } from 'node:module';
import fs from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createVerifiedPieceGather } from '../../src/storage/verified-piece-gather.js';
import { createVerifiedPieceGatherLane } from '../../src/storage/verified-piece-gather-lane.js';

const threads = createRequire(import.meta.url)('node:worker_threads'), OriginalWorker = threads.Worker;
const workerURL = new URL('../../src/storage/verified-piece-gather-worker.js', import.meta.url);
const sha = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const encode = value => new TextEncoder().encode(JSON.stringify(value));
const I = { reads: 0, readDone: 1, activeReads: 2, maxReads: 3, readError: 4, releaseReads: 5,
  closedFDs: 6, openedFDs: 7, syncReads: 8, hashes: 9, hashDone: 10, activeHashes: 11,
  maxHashes: 12, syncHashes: 13, enabled: 14, releaseHashes: 15, terminate: 16, failClose: 17, fstats: 18, lstats: 19,
  syncPauseAt: 20, syncPaused: 21, releaseSync: 22,
  syncReadPauseAt: 23, syncReadPaused: 24, releaseSyncRead: 25, readGate: 32, hashGate: 80 };

// Test-only preload: production workers receive no instrumentation options or modes.
// Native reads/digests still execute; only their issue/completion order is controlled.
const preload = new URL('data:text/javascript,' + encodeURIComponent(`
import fs from 'node:fs';
import crypto from 'node:crypto';
import { workerData } from 'node:worker_threads';
import { syncBuiltinESMExports } from 'node:module';
const S = new Int32Array(workerData.ioControls), I = ${JSON.stringify(I)};
const read = fs.read, readSync = fs.readSync, open = fs.openSync, close = fs.closeSync;
const fstat = fs.fstatSync, lstat = fs.lstatSync;
const owned = new Set(), approved = new Set(Object.values(workerData.files));
const hash = crypto.createHash, digest = crypto.webcrypto.subtle.digest.bind(crypto.webcrypto.subtle);
const increment = (active, peak) => { const n = Atomics.add(S, active, 1) + 1; Atomics.store(S, peak, Math.max(n, Atomics.load(S, peak))); };
const gated = (all, base, index, run) => {
  const poll = () => Atomics.load(S, all) || Atomics.load(S, base + index) ? run() : setTimeout(poll, 1);
  poll();
};
fs.openSync = (...args) => {
  const fd = open(...args);
  if (approved.has(args[0])) { owned.add(fd); Atomics.add(S, I.openedFDs, 1); }
  return fd;
};
fs.closeSync = fd => {
  if (owned.has(fd) && Atomics.compareExchange(S, I.failClose, 1, 0) === 1) throw new Error('Injected descriptor close failure');
  close(fd); if (owned.delete(fd)) Atomics.add(S, I.closedFDs, 1);
};
fs.fstatSync = (...args) => {
  if (owned.has(args[0]) && Atomics.load(S, I.enabled)) Atomics.add(S, I.fstats, 1);
  return fstat(...args);
};
fs.lstatSync = (...args) => {
  if (approved.has(args[0]) && Atomics.load(S, I.enabled)) Atomics.add(S, I.lstats, 1);
  return lstat(...args);
};
fs.readSync = (...args) => {
  if (!owned.has(args[0]) || !Atomics.load(S, I.enabled)) return readSync(...args);
  const index = Atomics.add(S, I.syncReads, 1) + 1;
  if (index === Atomics.load(S, I.syncReadPauseAt)) {
    Atomics.store(S, I.syncReadPaused, 1);
    if (Atomics.wait(S, I.releaseSyncRead, 0, 2000) === 'timed-out') throw new Error('Bounded synchronous read observation deadline');
  }
  return readSync(...args);
};
fs.read = (fd, bytes, offset, length, position, callback) => {
  const index = Atomics.add(S, I.reads, 1); increment(I.activeReads, I.maxReads);
  const done = (error, count, buffer) => {
    Atomics.sub(S, I.activeReads, 1); Atomics.add(S, I.readDone, 1); callback(error, count, buffer);
  };
  if (index === Atomics.load(S, I.readError)) { setImmediate(() => done(new Error('Injected positioned read failure'), 0, bytes)); return; }
  gated(I.releaseReads, I.readGate, index, () => read(fd, bytes, offset, length, position, done));
};
crypto.createHash = (...args) => {
  const current = hash(...args);
  if (!Atomics.load(S, I.enabled)) return current;
  const index = Atomics.add(S, I.syncHashes, 1) + 1, finish = current.digest.bind(current);
  current.digest = (...options) => {
    const result = finish(...options);
    if (index === Atomics.load(S, I.syncPauseAt)) {
      Atomics.store(S, I.syncPaused, 1);
      if (Atomics.wait(S, I.releaseSync, 0, 2000) === 'timed-out') throw new Error('Bounded synchronous hash observation deadline');
    }
    return result;
  };
  return current;
};
crypto.webcrypto.subtle.digest = (algorithm, bytes) => {
  if (!Atomics.load(S, I.enabled)) return digest(algorithm, bytes);
  const index = Atomics.add(S, I.hashes, 1); increment(I.activeHashes, I.maxHashes);
  return new Promise((resolve, reject) => gated(I.releaseHashes, I.hashGate, index, () => {
    digest(algorithm, bytes).then(resolve, reject).finally(() => {
      Atomics.sub(S, I.activeHashes, 1); Atomics.add(S, I.hashDone, 1);
    });
  }));
};
syncBuiltinESMExports();
`));

async function until(predicate) {
  const deadline = performance.now() + 2000;
  while (!predicate()) {
    assert(performance.now() < deadline, 'Bounded worker observation deadline');
    await new Promise(resolve => setTimeout(resolve, 2));
  }
}
async function fixture(maxPieceBytes = 8, wide = false) {
  const root = await fs.mkdtemp(join(tmpdir(), 'gather-parallel-'));
  const payloads = [Uint8Array.of(1, 2, 3, 4), Uint8Array.of(5, 6, 7, 8), Uint8Array.of(9, 10, 11, 12)];
  if (wide) for (let i = 0; i < payloads.length; i++) payloads[i] = Uint8Array.from([...payloads[i], 13 + i * 4, 14 + i * 4, 15 + i * 4, 16 + i * 4]);
  const names = ['gate.bin', 'up.bin', 'down.bin'];
  const files = Object.fromEntries(names.map(name => [name, join(root, name)]));
  await Promise.all(names.map((name, i) => fs.writeFile(files[name], payloads[i])));
  const manifestBytes = encode({ shards: names.map((filename, i) => ({ filename, size: payloads[i].length })) });
  const indexBytes = encode({ schema: 'doppler.verified-pieces/v1', manifestIdentity: sha(manifestBytes), maxPieceBytes,
    files: names.map((path, i) => ({ path, size: payloads[i].length,
      pieces: Array.from({ length: payloads[i].length / 4 }, (_, n) => ({ offset: n * 4, size: 4, identity: sha(payloads[i].subarray(n * 4, n * 4 + 4)) })) })) });
  const identities = Object.fromEntries(await Promise.all(names.map(async name => {
    const stat = await fs.stat(files[name], { bigint: true });
    return [name, { device: String(stat.dev), inode: String(stat.ino), size: String(stat.size) }];
  })));
  const state = new Int32Array(new SharedArrayBuffer(128 * 4)); state[I.readError] = -1;
  return { root, payloads, names, files, identities, state, maxPieceBytes,
    options: { manifestBytes, indexBytes, indexIdentity: sha(indexBytes), files, fileIdentities: identities,
      maxPartBytes: 12, memoryBudgetBytes: 12 + 2 * maxPieceBytes },
    ranges: names.map(path => ({ path, offset: 0, length: 4 })) };
}
async function direct(f) {
  const worker = new OriginalWorker(workerURL, { execArgv: ['--import', preload.href], workerData: {
    files: f.files, fileIdentities: f.identities, maxPartBytes: f.options.maxPartBytes,
    maxRanges: f.options.maxRanges, ioControls: f.state.buffer,
    metadata: { manifestBytes: f.options.manifestBytes, indexBytes: f.options.indexBytes, indexIdentity: f.options.indexIdentity },
  } });
  const messages = [], publications = []; let failure;
  worker.on('message', message => { messages.push(message); if (message.buffer) publications.push(message); });
  worker.on('error', error => { failure = error; });
  const take = async predicate => {
    await until(() => { if (failure) throw failure; return messages.some(predicate); });
    return messages.splice(messages.findIndex(predicate), 1)[0];
  };
  await take(message => message.ready); f.state[I.enabled] = 1;
  return { worker, publications, take,
    send(ranges = f.ranges, buffer = new ArrayBuffer(f.options.maxPartBytes)) {
      worker.postMessage({ ranges, length: ranges.reduce((n, range) => n + range.length, 0), buffer, batch: true }, [buffer]);
    },
    async receipt() { worker.postMessage({ receipt: true }); return (await take(message => message.receipt && !message.closed)).receipt; },
    async cleanup() { await worker.terminate(); await fs.rm(f.root, { recursive: true, force: true }); },
  };
}
async function publicHarness(f) {
  const S = f.state, workers = [], acks = [], outputs = [];
  threads.Worker = class extends OriginalWorker {
    constructor(url, options) {
      super(url, { ...options, execArgv: ['--import', preload.href], workerData: { ...options.workerData, ioControls: S.buffer } });
      workers.push(this);
      this.on('message', message => { if (message.closed) acks.push(message); if (message.buffer) outputs.push(message); });
    }
    terminate() { Atomics.add(S, I.terminate, 1); return super.terminate(); }
  };
  syncBuiltinESMExports();
  let gather;
  const cleanup = async () => {
    Atomics.store(S, I.releaseReads, 1); Atomics.store(S, I.releaseHashes, 1);
    Atomics.store(S, I.releaseSync, 1); Atomics.notify(S, I.releaseSync);
    Atomics.store(S, I.releaseSyncRead, 1); Atomics.notify(S, I.releaseSyncRead);
    await gather?.close().catch(() => {});
    threads.Worker = OriginalWorker; syncBuiltinESMExports();
    await Promise.allSettled(workers.map(worker => OriginalWorker.prototype.terminate.call(worker)));
    await fs.rm(f.root, { recursive: true, force: true });
  };
  try { gather = await createVerifiedPieceGather(f.options); S[I.enabled] = 1; }
  catch (error) { await cleanup(); throw error; }
  return { gather, workers, acks, outputs, cleanup,
    ranges: f.ranges.map(range => ({ shard: f.names.indexOf(range.path), offset: range.offset, length: range.length })) };
}

async function sameFileFixture() {
  const f = await fixture(), name = 'gate.bin', filename = f.files[name];
  const payload = Uint8Array.from(f.payloads.flatMap(bytes => [...bytes]));
  await fs.writeFile(filename, payload);
  const stat = await fs.stat(filename, { bigint: true });
  const manifestBytes = encode({ shards: [{ filename: name, size: payload.length }] });
  const indexBytes = encode({ schema: 'doppler.verified-pieces/v1', manifestIdentity: sha(manifestBytes), maxPieceBytes: f.maxPieceBytes,
    files: [{ path: name, size: payload.length, pieces: [0, 4, 8].map(offset => ({ offset, size: 4,
      identity: sha(payload.subarray(offset, offset + 4)) })) }] });
  f.files = { [name]: filename }; f.names = [name];
  f.identities = { [name]: { device: String(stat.dev), inode: String(stat.ino), size: String(stat.size) } };
  f.options = { ...f.options, manifestBytes, indexBytes, indexIdentity: sha(indexBytes), files: f.files, fileIdentities: f.identities };
  f.ranges = [0, 4, 8].map(offset => ({ path: name, offset, length: 4 }));
  return f;
}

test('one exact-piece batch uses positioned sync I/O while multi-piece batches retain actual async overlap', { timeout: 5000 }, async () => {
  const f = await fixture(), h = await direct(f), S = f.state;
  try {
    h.send([f.ranges[0]]);
    const single = await h.take(value => value.buffer);
    assert.equal(single.error, undefined);
    assert.deepEqual([...new Uint8Array(single.buffer, 0, 4)], [...f.payloads[0]]);
    assert.equal(S[I.syncReads], 1); assert.equal(S[I.reads], 0);
    assert.equal(S[I.syncHashes], 1, 'The synchronous read still authenticates its complete piece');
    assert.equal(S[I.lstats], 2); assert.equal(S[I.fstats], 3,
      'One before/after identity pair and the cold regular-file check, without redundant read guards');
    h.send(f.ranges, single.buffer);
    await until(() => S[I.reads] === 3);
    assert.equal(S[I.activeReads], 3); assert.equal(S[I.maxReads], 3);
    assert.equal(S[I.syncReads], 1, 'Multi-piece work must not regress to serialized synchronous I/O');
    Atomics.store(S, I.releaseReads, 1);
    const multi = await h.take(value => value.buffer);
    assert.equal(multi.error, undefined);
    assert.deepEqual([...new Uint8Array(multi.buffer)], f.payloads.flatMap(bytes => [...bytes]));
    assert.equal(S[I.syncHashes], 4);
    const receipt = await h.receipt();
    assert.equal(receipt.directPieceReads, 4); assert.equal(receipt.directPieceBytes, 16);
    assert.equal(receipt.peakDigestCopyBytes, 0); assert.equal(receipt.peakGenericScratchBytes, 0);
  } finally { await h.cleanup(); }
});

test('single exact-piece sync batches retain complete SHA, EOF rejection and post-hash file or symlink identity guards', { timeout: 5000 }, async () => {
  for (const change of ['corrupt', 'file', 'symlink', 'eof']) {
    const f = await fixture(), h = await direct(f), S = f.state;
    const filename = f.files['gate.bin'], saved = filename + '.saved';
    try {
      if (change === 'corrupt') await fs.writeFile(filename, new Uint8Array(4));
      else if (change === 'eof') Atomics.store(S, I.syncReadPauseAt, 1);
      else Atomics.store(S, I.syncPauseAt, 1);
      h.send([f.ranges[0]]);
      if (change === 'eof') {
        await until(() => S[I.syncReadPaused] === 1);
        await fs.truncate(filename, 1);
        Atomics.store(S, I.releaseSyncRead, 1); Atomics.notify(S, I.releaseSyncRead);
      } else if (change !== 'corrupt') {
        await until(() => S[I.syncPaused] === 1);
        assert.equal(h.publications.length, 0);
        await fs.rename(filename, saved);
        if (change === 'file') await fs.writeFile(filename, f.payloads[0]);
        else await fs.symlink(saved, filename);
        Atomics.store(S, I.releaseSync, 1); Atomics.notify(S, I.releaseSync);
      }
      const failed = await h.take(value => value.buffer);
      assert.match(failed.error, change === 'corrupt' ? /integrity/ : change === 'eof' ? /Incomplete authenticated piece/ : /identity changed/);
      assert(new Uint8Array(failed.buffer).every(byte => byte === 0));
      assert.equal(S[I.reads], 0, 'A single complete-piece job never enters the shared async pool');
      if (change === 'file' || change === 'symlink') {
        await fs.unlink(filename); await fs.rename(saved, filename);
      } else await fs.writeFile(filename, f.payloads[0]);
      h.send([f.ranges[0]], failed.buffer);
      const recovered = await h.take(value => value.buffer);
      assert.equal(recovered.error, undefined);
      assert.deepEqual([...new Uint8Array(recovered.buffer, 0, 4)], [...f.payloads[0]]);
    } finally {
      Atomics.store(S, I.releaseSync, 1); Atomics.notify(S, I.releaseSync);
      Atomics.store(S, I.releaseSyncRead, 1); Atomics.notify(S, I.releaseSyncRead);
      await h.cleanup();
    }
  }
});

test('cancel and close wait for single-piece positioned sync I/O to relinquish its own worker and descriptor', { timeout: 5000 }, async () => {
  for (const action of ['cancel', 'close']) {
    const f = await fixture(), S = f.state, h = await publicHarness(f);
    try {
      Atomics.store(S, I.syncReadPauseAt, 1);
      const controller = new AbortController();
      const pending = h.gather.rangeLease.readMany([h.ranges[0]], controller.signal);
      let settled = false;
      const result = pending.then(value => { settled = true; return { value }; }, error => { settled = true; return { error }; });
      await until(() => S[I.syncReadPaused] === 1);
      let closed = false, closing;
      if (action === 'cancel') controller.abort(Error('cancel single synchronous piece'));
      else closing = h.gather.close().then(() => { closed = true; });
      await Promise.resolve();
      assert.equal(S[I.closedFDs], 0); assert.equal(S[I.terminate], 0); assert.equal(h.acks.length, 0);
      if (action === 'cancel') assert.equal(settled, false, 'Cancellation cannot interrupt a native synchronous read');
      else assert.equal(closed, false, 'Retirement acknowledgement must wait for the worker event loop');
      Atomics.store(S, I.releaseSyncRead, 1); Atomics.notify(S, I.releaseSyncRead);
      const outcome = await result;
      outcome.value?.release();
      assert.match(outcome.error?.message ?? '', action === 'cancel' ? /cancel single synchronous piece/ : /Gather reader closed/);
      if (action === 'cancel') {
        const recovered = await h.gather.rangeLease.readMany([h.ranges[0]], new AbortController().signal);
        assert.deepEqual([...recovered.bytes], [...f.payloads[0]]); recovered.release();
        await h.gather.close();
      } else await closing;
      assert.equal(S[I.openedFDs], S[I.closedFDs]);
      assert.equal(S[I.reads], 0); assert.equal(S[I.terminate], 1);
      assert.equal(h.acks.length, 1, 'Close acknowledges settled work before termination');
    } finally { await h.cleanup(); }
  }
});

test('whole-expert identity guards run once per distinct file at both request boundaries, never across jobs', { timeout: 5000 }, async () => {
  const f = await sameFileFixture(), h = await direct(f), S = f.state;
  try {
    Atomics.store(S, I.releaseReads, 1); Atomics.store(S, I.releaseHashes, 1);
    h.send(); const first = await h.take(value => value.buffer);
    assert.equal(first.error, undefined);
    assert.equal(S[I.openedFDs], 1);
    assert.equal(S[I.lstats], 2, 'One path guard before all reads and one after all digests');
    assert.equal(S[I.fstats], 3, 'Two identity guards plus the first descriptor regular-file check');
    assert.equal(S[I.syncHashes], 3, 'Every complete piece retains its own SHA');
    Atomics.store(S, I.lstats, 0); Atomics.store(S, I.fstats, 0);
    h.send(f.ranges, first.buffer); const second = await h.take(value => value.buffer);
    assert.equal(second.error, undefined);
    assert.equal(S[I.lstats], 2); assert.equal(S[I.fstats], 2, 'No cross-job identity cache');
    assert.equal(S[I.syncHashes], 6); assert.equal(S[I.hashes], 0);
    assert.deepEqual([...new Uint8Array(second.buffer)], f.payloads.flatMap(bytes => [...bytes]));
  } finally { await h.cleanup(); }
});

test('same-file batches reject mutation during reads and path replacement after hashing before publication', { timeout: 5000 }, async () => {
  for (const phase of ['read', 'hash']) {
    const f = await sameFileFixture(), h = await direct(f), S = f.state, filename = f.files['gate.bin'];
    try {
      if (phase === 'hash') Atomics.store(S, I.syncPauseAt, 1);
      h.send(); await until(() => S[I.reads] === 3);
      assert.equal(h.publications.length, 0);
      assert.equal(S[I.lstats], 1, 'All same-file reads share one job admission guard');
      assert.equal(S[I.fstats], 2, 'Admission includes the cold regular-file descriptor check');
      if (phase === 'read') await fs.writeFile(filename, new Uint8Array(12));
      if (phase === 'hash') {
        Atomics.store(S, I.readGate + 2, 1); await until(() => S[I.syncPaused] === 1);
        await fs.rename(filename, filename + '.saved'); await fs.writeFile(filename, new Uint8Array(12));
        assert.equal(S[I.syncHashes], 1);
        assert.equal(h.publications.length, 0, 'One authenticated piece never releases remaining native read owners');
        assert.equal(S[I.lstats], 1, 'The final identity guard waits for all native reads and hashes');
        Atomics.store(S, I.releaseSync, 1); Atomics.notify(S, I.releaseSync);
      }
      Atomics.store(S, I.releaseReads, 1);
      const failed = await h.take(value => value.buffer);
      assert.match(failed.error, phase === 'read' ? /integrity/ : /identity changed/);
      assert(new Uint8Array(failed.buffer).every(byte => byte === 0));
      assert.equal(S[I.lstats], 2, 'Even rejected jobs complete their final identity guard after all siblings');
      assert.equal(S[I.activeReads], 0); assert.equal(S[I.activeHashes], 0);
      if (phase === 'hash') { await fs.unlink(filename); await fs.rename(filename + '.saved', filename); }
      else await fs.writeFile(filename, Uint8Array.from(f.payloads.flatMap(bytes => [...bytes])));
      h.send(f.ranges, failed.buffer); const recovered = await h.take(value => value.buffer);
      assert.equal(recovered.error, undefined);
    } finally { Atomics.store(S, I.releaseSync, 1); Atomics.notify(S, I.releaseSync); await h.cleanup(); }
  }
});

test('batch identity capability requires actual complete pinned file bindings, never copied assertions', { timeout: 5000 }, async () => {
  for (const bound of [true, false]) {
    const f = await fixture();
    if (!bound) f.options = { ...f.options, fileIdentities: undefined,
      fileIdentitiesBound: true, batchFileIdentityChecks: 'before-read-after-auth/v1' };
    const h = await publicHarness(f);
    try {
      assert.equal(h.gather.rangeLease.fileIdentitiesBound, bound);
      assert.equal(h.gather.rangeLease.batchFileIdentityChecks, bound ? 'before-read-after-auth/v1' : undefined);
      const lease = await h.gather.rangeLease.readMany([{ shard: 0, offset: 1, length: 2 }], new AbortController().signal);
      assert.deepEqual([...lease.bytes], [2, 3]); lease.release();
    } finally { await h.cleanup(); }
  }
  const f = await fixture();
  try {
    await assert.rejects(createVerifiedPieceGather({ ...f.options,
      fileIdentities: { 'gate.bin': f.identities['gate.bin'] } }), /identit|bind/i);
  } finally { await fs.rm(f.root, { recursive: true, force: true }); }
});

test('single leases reject matching-content path swaps after authentication and recover without publishing bytes', { timeout: 5000 }, async () => {
  for (const length of [4, 2]) for (const replacement of ['file', 'symlink']) {
    const f = await fixture(), h = await publicHarness(f), S = f.state;
    const filename = f.files['gate.bin'], saved = filename + '.saved', offset = length === 4 ? 0 : 1;
    try {
      const empty = await h.gather.readLease('gate.bin', 4, 0);
      assert.equal(empty.bytes.byteLength, 0); empty.release();
      assert.equal(S[I.openedFDs], 0, 'An empty read needs no admitted descriptor');
      h.outputs.length = 0;
      Atomics.store(S, I.syncPauseAt, 1);
      const pending = h.gather.readLease('gate.bin', offset, length);
      const outcome = pending.then(value => ({ value }), error => ({ error }));
      await until(() => S[I.syncPaused] === 1);
      assert.equal(h.outputs.length, 0, 'Authenticated bytes remain worker-owned before publication');
      await fs.rename(filename, saved);
      if (replacement === 'file') await fs.writeFile(filename, f.payloads[0]);
      else await fs.symlink(saved, filename);
      Atomics.store(S, I.releaseSync, 1); Atomics.notify(S, I.releaseSync);
      const result = await outcome;
      result.value?.release();
      assert.match(result.error?.message ?? '', /identity changed/, 'A different approved-path binding must fail after hashing');
      assert.equal(h.outputs.length, 1);
      assert.ok(new Uint8Array(h.outputs[0].buffer).every(byte => byte === 0), 'Failed output is cleared before return to its lane');
      await fs.unlink(filename); await fs.rename(saved, filename);
      const recovered = await h.gather.readLease('gate.bin', offset, length);
      assert.deepEqual([...recovered.bytes], [...f.payloads[0].slice(offset, offset + length)]);
      recovered.release();
      await h.gather.close();
      assert.equal(S[I.closedFDs], S[I.openedFDs], 'Restored request and close settle every owned descriptor');
    } finally { await h.cleanup(); }
  }
});

test('partial batches revalidate earlier file bindings after the last hash before any publication', { timeout: 5000 }, async () => {
  for (const replacement of ['file', 'symlink']) {
    const f = await fixture(), h = await direct(f), S = f.state, filename = f.files['gate.bin'];
    try {
      Atomics.store(S, I.syncPauseAt, 3);
      h.send(f.ranges.map(range => ({ ...range, offset: 1, length: 2 })));
      await until(() => S[I.syncPaused] === 1);
      assert.equal(h.publications.length, 0, 'Even authenticated outputs remain worker-owned');
      await fs.rename(filename, filename + '.saved');
      if (replacement === 'file') await fs.writeFile(filename, f.payloads[0]);
      else await fs.symlink(filename + '.saved', filename);
      Atomics.store(S, I.releaseSync, 1); Atomics.notify(S, I.releaseSync);
      const failed = await h.take(value => value.buffer);
      assert.match(failed.error, /identity changed/);
      assert(new Uint8Array(failed.buffer).every(byte => byte === 0));
      assert.equal(S[I.syncHashes], 3, 'Each complete intersecting piece was authenticated');
      await fs.unlink(filename); await fs.rename(filename + '.saved', filename);
      h.send(f.ranges.map(range => ({ ...range, offset: 1, length: 2 })), failed.buffer);
      const recovered = await h.take(value => value.buffer);
      assert.equal(recovered.error, undefined);
      assert.deepEqual([...new Uint8Array(recovered.buffer, 0, 6)], [2, 3, 6, 7, 10, 11]);
    } finally {
      Atomics.store(S, I.releaseSync, 1); Atomics.notify(S, I.releaseSync);
      await h.cleanup();
    }
  }
});

async function multiExpertFixture() {
  const f = await fixture();
  f.options = { ...f.options, maxRanges: 9, maxPartBytes: 36, memoryBudgetBytes: 52 };
  f.ranges = [0, 1, 0, 2, 1, 2, 0, 0, 2].map(index => ({ path: f.names[index], offset: 0, length: 4 }));
  return f;
}

test('larger batches require an explicit bounded maxRanges before any worker admission', { timeout: 5000 }, async () => {
  const f = await fixture(); let started = 0;
  threads.Worker = class extends OriginalWorker { constructor(...args) { started++; super(...args); } };
  syncBuiltinESMExports();
  try {
    for (const maxRanges of [0, -1, 1.5, 33, Infinity, NaN, Number.MAX_SAFE_INTEGER + 1, '9', null, true]) {
      for (const create of [createVerifiedPieceGather, createVerifiedPieceGatherLane]) {
        await assert.rejects(create({ ...f.options, maxRanges }), /maxRanges/);
      }
    }
    assert.equal(started, 0);
    const reader = await createVerifiedPieceGather(f.options);
    try {
      assert.equal(reader.rangeLease.maxRanges, 3);
      await assert.rejects(reader.rangeLease.readMany(Array(4).fill({ shard: 0, offset: 0, length: 1 }),
        new AbortController().signal), /ranges bound/);
    } finally { await reader.close(); }
  } finally {
    threads.Worker = OriginalWorker; syncBuiltinESMExports();
    await fs.rm(f.root, { recursive: true, force: true });
  }
});

test('nine-range batches retain output and full-piece SHA bounds across repeated and distinct files', { timeout: 5000 }, async () => {
  const f = await multiExpertFixture(), h = await publicHarness(f), S = f.state;
  try {
    assert.equal(h.gather.rangeLease.maxRanges, 9); assert.equal(h.gather.rangeLease.memoryBytes, 52);
    assert.equal(h.gather.rangeLease.batchFileIdentityChecks, 'before-read-after-auth/v1');
    Atomics.store(S, I.releaseReads, 1);
    const ranges = f.ranges.map(range => ({ shard: f.names.indexOf(range.path), offset: range.offset, length: range.length }));
    const lease = await h.gather.rangeLease.readMany(ranges, new AbortController().signal);
    assert.deepEqual([...lease.bytes], ranges.flatMap(range => [...f.payloads[range.shard]]));
    assert.equal(S[I.reads], 9); assert.equal(S[I.syncHashes], 9); assert.equal(S[I.hashes], 0);
    assert.equal(S[I.lstats], 6, 'One pair of identity guards for each of three files');
    lease.release();
    const receipt = await h.gather.getReceipt();
    assert.equal(receipt.batchRanges, 9); assert.equal(receipt.directPieceReads, 9);
    assert.equal(receipt.peakDigestCopyBytes, 0, 'Outputs exceeding two pieces retain synchronous SHA');
    for (const invalid of [Array(10).fill(ranges[0]), ranges.map(range => ({ ...range, length: 5 })),
      [{ ...ranges[0], length: 37 }]]) {
      await assert.rejects(h.gather.rangeLease.readMany(invalid, new AbortController().signal), /bound/);
    }
    const partial = ranges.map(range => ({ ...range, offset: 1, length: 2 }));
    const partialLease = await h.gather.rangeLease.readMany(partial, new AbortController().signal);
    assert.deepEqual([...partialLease.bytes], partial.flatMap(range => [...f.payloads[range.shard].subarray(1, 3)]));
    partialLease.release();
    const corrupt = f.payloads[0].slice(); corrupt[0] ^= 255;
    await fs.writeFile(f.files[f.names[0]], corrupt);
    await assert.rejects(h.gather.rangeLease.readMany(partial, new AbortController().signal), /integrity/);
    assert(new Uint8Array(h.outputs.at(-1).buffer).every(byte => byte === 0));
    await fs.writeFile(f.files[f.names[0]], f.payloads[0]);
    const recovered = await h.gather.rangeLease.readMany(partial, new AbortController().signal); recovered.release();
  } finally { await h.cleanup(); }
});

test('cancelled nine-range jobs drain every sibling native read before recycling or recovery', { timeout: 5000 }, async () => {
  const f = await multiExpertFixture(), h = await publicHarness(f), S = f.state;
  try {
    Atomics.store(S, I.releaseHashes, 1);
    const controller = new AbortController();
    let settled = false;
    const pending = h.gather.rangeLease.readMany(f.ranges.map(range => ({ shard: f.names.indexOf(range.path),
      offset: range.offset, length: range.length })), controller.signal);
    const rejected = assert.rejects(pending, /cancel nine/).finally(() => { settled = true; });
    await until(() => S[I.reads] === 9); controller.abort(new Error('cancel nine'));
    for (let index = 0; index < 8; index++) Atomics.store(S, I.readGate + index, 1);
    await until(() => S[I.readDone] === 8);
    assert.equal(settled, false); assert.equal(h.outputs.length, 0);
    assert.equal(S[I.activeReads], 1);
    Atomics.store(S, I.releaseReads, 1); await rejected;
    assert.equal(S[I.activeReads], 0); assert.equal(S[I.syncHashes], 9);
    assert(new Uint8Array(h.outputs.at(-1).buffer).every(byte => byte === 0));
    const recovered = await h.gather.rangeLease.readMany([{ shard: 1, offset: 0, length: 4 }], new AbortController().signal);
    assert.deepEqual([...recovered.bytes], [...f.payloads[1]]); recovered.release();
  } finally { await h.cleanup(); }
});

test('three direct reads overlap and hash their owned outputs without snapshots before one ordered transfer', { timeout: 5000 }, async () => {
  const f = await fixture(), h = await direct(f), S = f.state;
  try {
    h.send(); await until(() => S[I.reads] === 3);
    assert.equal(S[I.readDone], 0); assert.equal(S[I.maxReads], 3);
    Atomics.store(S, I.readGate + 2, 1); await until(() => S[I.syncHashes] === 1);
    assert.equal(S[I.readDone], 1); assert.equal(h.publications.length, 0);
    const active = await h.receipt();
    assert.equal(active.activeDigestCopyBytes, 0); assert.equal(active.peakDigestCopyBytes, 0);
    assert.equal(S[I.activeReads], 2, 'Authenticated first output remains owned until both native siblings settle');
    Atomics.store(S, I.releaseReads, 1);
    const message = await h.take(value => value.buffer);
    assert.equal(message.error, undefined); assert.deepEqual([...new Uint8Array(message.buffer)], f.payloads.flatMap(bytes => [...bytes]));
    const receipt = await h.receipt();
    assert.equal(receipt.activeDigestCopyBytes, 0); assert.equal(receipt.directPieceReads, 3);
    assert.equal(receipt.peakDigestCopyBytes, 0); assert.equal(S[I.syncHashes], 3); assert.equal(S[I.hashes], 0);
  } finally { await h.cleanup(); }
});

test('one rejected read waits for late sibling writes and hashes before zero, then recovers', { timeout: 5000 }, async () => {
  const f = await fixture(), h = await direct(f), S = f.state;
  try {
    Atomics.store(S, I.readError, 1); Atomics.store(S, I.syncPauseAt, 1); h.send();
    await until(() => S[I.reads] === 3 && S[I.readDone] === 1);
    assert.equal(h.publications.length, 0);
    Atomics.store(S, I.readGate, 1); await until(() => S[I.syncPaused] === 1);
    assert.equal(h.publications.length, 0, 'Read rejection cannot zero or transfer while a synchronous hash still owns the output');
    Atomics.store(S, I.releaseSync, 1); Atomics.notify(S, I.releaseSync);
    assert.equal(h.publications.length, 0, 'Read rejection must await the late native sibling even after another piece hashes');
    assert.equal(S[I.activeReads], 1);
    Atomics.store(S, I.releaseReads, 1);
    const failed = await h.take(value => value.buffer);
    assert.match(failed.error, /positioned read failure/); assert(new Uint8Array(failed.buffer).every(byte => byte === 0));
    assert.equal(S[I.activeReads], 0); assert.equal(S[I.activeHashes], 0);
    assert.equal(S[I.syncHashes], 2); assert.equal(S[I.hashes], 0);
    assert.equal((await h.receipt()).activeDigestCopyBytes, 0);
    Atomics.store(S, I.readError, -1); h.send(f.ranges, failed.buffer);
    const recovered = await h.take(value => value.buffer);
    assert.deepEqual([...new Uint8Array(recovered.buffer)], f.payloads.flatMap(bytes => [...bytes]));
  } finally { Atomics.store(S, I.releaseSync, 1); Atomics.notify(S, I.releaseSync); await h.cleanup(); }
});

test('owned-output SHA detects corrupt content and post-hash inode swaps without publishing a prefix', { timeout: 5000 }, async () => {
  for (const change of ['corrupt', 'swap']) {
    const f = await fixture(), h = await direct(f), S = f.state, filename = f.files['down.bin'];
    try {
      if (change === 'corrupt') await fs.writeFile(filename, Uint8Array.of(0, 0, 0, 0));
      if (change === 'swap') Atomics.store(S, I.syncPauseAt, 3);
      Atomics.store(S, I.releaseReads, 1); h.send();
      if (change === 'swap') {
        await until(() => S[I.syncPaused] === 1);
        assert.equal(h.publications.length, 0);
        await fs.rename(filename, filename + '.saved'); await fs.writeFile(filename, f.payloads[2]);
        Atomics.store(S, I.releaseSync, 1); Atomics.notify(S, I.releaseSync);
      }
      const failed = await h.take(value => value.buffer);
      assert.match(failed.error, change === 'corrupt' ? /integrity/ : /identity changed/);
      assert(new Uint8Array(failed.buffer).every(byte => byte === 0));
      assert.equal((await h.receipt()).activeDigestCopyBytes, 0);
      if (change === 'swap') { await fs.unlink(filename); await fs.rename(filename + '.saved', filename); }
      else await fs.writeFile(filename, f.payloads[2]);
      h.send(f.ranges, failed.buffer); const recovered = await h.take(value => value.buffer);
      assert.equal(recovered.error, undefined);
      assert.deepEqual([...new Uint8Array(recovered.buffer)], f.payloads.flatMap(bytes => [...bytes]));
    } finally { Atomics.store(S, I.releaseSync, 1); Atomics.notify(S, I.releaseSync); await h.cleanup(); }
  }
});

test('larger whole-piece batches hash owned outputs and partial ranges retain serialized authenticated scratch', { timeout: 5000 }, async () => {
  for (const partial of [false, true]) {
    const f = await fixture(4), h = await direct(f), S = f.state;
    try {
      if (partial) h.send([f.ranges[0], { ...f.ranges[1], offset: 1, length: 2 }]);
      else { Atomics.store(S, I.releaseReads, 1); h.send(); }
      const output = await h.take(value => value.buffer);
      assert.equal(output.error, undefined);
      assert.equal(S[I.reads], partial ? 0 : 3);
      assert(S[I.syncReads] >= (partial ? 2 : 0));
      if (!partial) { assert.equal(S[I.hashes], 0); assert.equal(S[I.syncHashes], 3); }
      const receipt = await h.receipt();
      assert.equal(receipt.peakDigestCopyBytes, 0); assert.equal(S[I.hashes], 0, 'Generic worker verification does not allocate native async digest input copies');
      assert.equal(receipt.activeGenericScratchBytes, 0);
      assert.equal(receipt.peakGenericScratchBytes, partial ? f.maxPieceBytes : 0);
    } finally { await h.cleanup(); }
  }
});

test('serialized partial ranges authenticate every complete piece with one owned scratch and recover after corruption', { timeout: 5000 }, async () => {
  const f = await fixture(8, true), h = await direct(f), filename = f.files['gate.bin'];
  const ranges = [{ path: 'gate.bin', offset: 2, length: 4 }, { path: 'up.bin', offset: 1, length: 2 }];
  try {
    h.send(ranges); const first = await h.take(value => value.buffer);
    assert.equal(first.error, undefined); assert.deepEqual([...new Uint8Array(first.buffer, 0, first.length)], [3, 4, 13, 14, 6, 7]);
    const receipt = await h.receipt();
    assert.equal(receipt.requestedBytes, 6); assert.equal(receipt.verifiedBytes, 12);
    assert.equal(receipt.ranges.length, 3); assert.equal(receipt.peakGenericScratchBytes, 8);
    assert.equal(receipt.activeGenericScratchBytes, 0); assert.equal(receipt.peakDigestCopyBytes, 0);
    assert.equal(f.state[I.reads], 0); assert.equal(f.state[I.hashes], 0);
    const corrupted = f.payloads[0].slice(); corrupted[0] ^= 255;
    await fs.writeFile(filename, corrupted); h.send(ranges, first.buffer);
    const failed = await h.take(value => value.buffer);
    assert.match(failed.error, /integrity/); assert(new Uint8Array(failed.buffer).every(byte => byte === 0));
    assert.equal((await h.receipt()).activeGenericScratchBytes, 0);
    await fs.writeFile(filename, f.payloads[0]); h.send(ranges, failed.buffer);
    const recovered = await h.take(value => value.buffer);
    assert.equal(recovered.error, undefined); assert.deepEqual([...new Uint8Array(recovered.buffer, 0, recovered.length)], [3, 4, 13, 14, 6, 7]);
  } finally { await h.cleanup(); }
});

test('lane close rejects immediately but awaits owned native reads or synchronous hash and closed descriptors before termination', { timeout: 10000 }, async () => {
  for (const phase of ['read', 'hash']) {
    const f = await fixture(), S = f.state, h = await publicHarness(f), { gather, acks, outputs } = h;
    try {
      if (phase === 'hash') {
        Atomics.store(S, I.syncPauseAt, 1); Atomics.store(S, I.releaseReads, 1);
      }
      const pending = gather.rangeLease.readMany(h.ranges, new AbortController().signal);
      await until(() => phase === 'read' ? S[I.reads] === 3 : S[I.syncPaused] === 1);
      const rejected = assert.rejects(pending, error => {
        assert.equal(error.message, 'Gather reader closed.'); return true;
      });
      let settled = false; const closing = gather.close().then(() => { settled = true; });
      await rejected;
      assert.equal(settled, false); assert.equal(S[I.terminate], 0); assert.equal(acks.length, 0);
      if (phase === 'read') {
        Atomics.store(S, I.readGate + 2, 1);
        await until(() => S[I.readDone] === 1);
      } else {
        assert.equal(S[I.syncHashes], 1, 'Close cannot retire the worker while its completed digest still owns publication');
        assert.equal(S[I.closedFDs], 0);
      }
      assert.equal(settled, false); assert.equal(acks.length, 0);
      Atomics.store(S, I.releaseReads, 1); Atomics.store(S, I.releaseHashes, 1);
      Atomics.store(S, I.releaseSync, 1); Atomics.notify(S, I.releaseSync);
      await closing;
      assert.equal(S[I.readDone], 3); assert.equal(S[I.activeReads], 0); assert.equal(S[I.activeHashes], 0);
      assert.equal(S[I.openedFDs], 3); assert.equal(S[I.closedFDs], 3);
      assert.equal(acks.length, 1); assert.equal(acks[0].receipt.activeDigestCopyBytes, 0);
      assert.equal(acks[0].receipt.peakDigestCopyBytes, 0); assert.equal(S[I.hashes], 0);
      assert.equal(S[I.terminate], 1);
      assert(outputs.every(message => new Uint8Array(message.buffer).every(byte => byte === 0)));
    } finally { await h.cleanup(); }
  }
});

test('cancellation awaits owned jobs without publication and the same lane recovers', { timeout: 5000 }, async () => {
  const f = await fixture(), S = f.state, h = await publicHarness(f);
  try {
    Atomics.store(S, I.releaseReads, 1);
    Atomics.store(S, I.syncPauseAt, 1);
    const controller = new AbortController();
    const pending = h.gather.rangeLease.readMany(h.ranges, controller.signal);
    await until(() => S[I.syncPaused] === 1);
    controller.abort(new Error('Cancelled parallel batch'));
    const rejected = assert.rejects(pending, /Cancelled parallel batch/);
    assert.equal(h.outputs.length, 0);
    Atomics.store(S, I.releaseSync, 1); Atomics.notify(S, I.releaseSync); await rejected;
    assert.equal((await h.gather.getReceipt()).activeDigestCopyBytes, 0);
    assert.equal(S[I.activeReads], 0); assert.equal(S[I.syncHashes], 3); assert.equal(S[I.hashes], 0);
    assert(new Uint8Array(h.outputs.at(-1).buffer).every(byte => byte === 0));
    const recovered = await h.gather.rangeLease.readMany(h.ranges, new AbortController().signal);
    assert.deepEqual([...recovered.bytes], f.payloads.flatMap(bytes => [...bytes])); recovered.release();
  } finally { await h.cleanup(); }
});

test('partial fallback cancellation and close retain the original errors and settle scratch ownership', { timeout: 5000 }, async () => {
  const f = await fixture(8, true), h = await publicHarness(f);
  const ranges = [{ shard: 0, offset: 2, length: 4 }, { shard: 1, offset: 1, length: 2 }];
  try {
    const controller = new AbortController();
    const pending = h.gather.rangeLease.readMany(ranges, controller.signal);
    controller.abort(new Error('Cancelled partial fallback'));
    await assert.rejects(pending, error => { assert.equal(error.message, 'Cancelled partial fallback'); return true; });
    const receipt = await h.gather.getReceipt();
    assert.equal(receipt.activeGenericScratchBytes, 0); assert.equal(receipt.peakGenericScratchBytes, 8);
    assert.equal(receipt.activeDigestCopyBytes, 0); assert.equal(receipt.peakDigestCopyBytes, 0);
    const recovered = await h.gather.rangeLease.readMany(ranges, new AbortController().signal);
    assert.deepEqual([...recovered.bytes], [3, 4, 13, 14, 6, 7]); recovered.release();
    const closingRead = h.gather.rangeLease.readMany(ranges, new AbortController().signal);
    const rejected = assert.rejects(closingRead, error => { assert.equal(error.message, 'Gather reader closed.'); return true; });
    await h.gather.close(); await rejected;
    assert.equal(h.acks.at(-1).receipt.activeGenericScratchBytes, 0);
    assert.equal(f.state[I.reads], 0); assert.equal(f.state[I.hashes], 0);
    assert.equal(f.state[I.openedFDs], f.state[I.closedFDs]);
  } finally { await h.cleanup(); }
});

test('descriptor retirement failure remains owned and retryable before thread termination', { timeout: 5000 }, async () => {
  const f = await fixture(), S = f.state, h = await publicHarness(f);
  try {
    Atomics.store(S, I.releaseReads, 1); Atomics.store(S, I.releaseHashes, 1);
    const lease = await h.gather.rangeLease.readMany(h.ranges, new AbortController().signal); lease.release();
    Atomics.store(S, I.failClose, 1);
    await assert.rejects(h.gather.close(), /descriptor retirement failed/);
    assert.equal(S[I.terminate], 0); assert.equal(S[I.closedFDs], 2); assert(h.workers[0].threadId > 0);
    await h.gather.close();
    assert.equal(S[I.closedFDs], 3); assert.equal(S[I.terminate], 1);
    assert.equal(h.acks.at(-1).receipt.activeDigestCopyBytes, 0);
  } finally { await h.cleanup(); }
});

test('missing worker retirement acknowledgement rejects without hanging or claiming settled I/O', { timeout: 5000 }, async () => {
  const f = await fixture(), h = await publicHarness(f);
  try {
    await OriginalWorker.prototype.terminate.call(h.workers[0]);
    await assert.rejects(h.gather.close(), /before acknowledging settled reads/);
    await assert.rejects(h.gather.close(), /before acknowledging settled reads/);
    assert.equal(h.acks.length, 0);
  } finally { await h.cleanup(); }
});

test('four concurrent lanes authenticate all owned outputs with zero digest snapshots inside the unchanged reservation', { timeout: 5000 }, async () => {
  const f = await fixture(), S = f.state;
  f.options.concurrency = 4; f.options.memoryBudgetBytes *= 4;
  const h = await publicHarness(f);
  try {
    const reads = Array.from({ length: 4 }, () => h.gather.rangeLease.readMany(h.ranges, new AbortController().signal));
    await until(() => S[I.reads] === 12);
    assert.equal(S[I.activeReads], 12); assert.equal(S[I.maxReads], 12);
    Atomics.store(S, I.releaseReads, 1);
    for (const lease of await Promise.all(reads)) {
      assert(Object.isFrozen(lease.timings), 'Worker timings survive the lane and pool as frozen scalars');
      assert.deepEqual(Object.keys(lease.timings).sort(), ['copyMs', 'readWaitMs', 'roundTripMs', 'verificationMs', 'workerWallMs']);
      assert(Object.values(lease.timings).every(value => Number.isFinite(value) && value >= 0));
      lease.release();
    }
    const receipt = await h.gather.getReceipt();
    assert.equal(receipt.activeDigestCopyBytes, 0); assert.equal(receipt.peakDigestCopyBytes, 0);
    assert.equal(S[I.syncHashes], 12); assert.equal(S[I.hashes], 0);
    assert.equal(receipt.directPieceReads, 12); assert.equal(receipt.directPieceBytes, 48);
    assert.equal(h.gather.rangeLease.memoryBytes, f.options.memoryBudgetBytes);
  } finally { await h.cleanup(); }
});
