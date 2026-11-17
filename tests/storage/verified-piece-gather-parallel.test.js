import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createHash } from 'node:crypto';
import { createRequire, syncBuiltinESMExports } from 'node:module';
import fs from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createVerifiedPieceGather } from '../../src/storage/verified-piece-gather.js';

const threads = createRequire(import.meta.url)('node:worker_threads'), OriginalWorker = threads.Worker;
const workerURL = new URL('../../src/storage/verified-piece-gather-worker.js', import.meta.url);
const sha = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const encode = value => new TextEncoder().encode(JSON.stringify(value));
const I = { reads: 0, readDone: 1, activeReads: 2, maxReads: 3, readError: 4, releaseReads: 5,
  closedFDs: 6, openedFDs: 7, syncReads: 8, hashes: 9, hashDone: 10, activeHashes: 11,
  maxHashes: 12, syncHashes: 13, enabled: 14, releaseHashes: 15, terminate: 16, failClose: 17,
  readGate: 32, hashGate: 80 };

// Test-only preload: production workers receive no instrumentation options or modes.
// Native reads/digests still execute; only their issue/completion order is controlled.
const preload = new URL('data:text/javascript,' + encodeURIComponent(`
import fs from 'node:fs';
import crypto from 'node:crypto';
import { workerData } from 'node:worker_threads';
import { syncBuiltinESMExports } from 'node:module';
const S = new Int32Array(workerData.ioControls), I = ${JSON.stringify(I)};
const read = fs.read, readSync = fs.readSync, open = fs.openSync, close = fs.closeSync;
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
fs.readSync = (...args) => { Atomics.add(S, I.syncReads, 1); return readSync(...args); };
fs.read = (fd, bytes, offset, length, position, callback) => {
  const index = Atomics.add(S, I.reads, 1); increment(I.activeReads, I.maxReads);
  const done = (error, count, buffer) => {
    Atomics.sub(S, I.activeReads, 1); Atomics.add(S, I.readDone, 1); callback(error, count, buffer);
  };
  if (index === Atomics.load(S, I.readError)) { setImmediate(() => done(new Error('Injected positioned read failure'), 0, bytes)); return; }
  gated(I.releaseReads, I.readGate, index, () => read(fd, bytes, offset, length, position, done));
};
crypto.createHash = (...args) => { if (Atomics.load(S, I.enabled)) Atomics.add(S, I.syncHashes, 1); return hash(...args); };
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
    files: f.files, fileIdentities: f.identities, maxPartBytes: 12, ioControls: f.state.buffer,
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
    send(ranges = f.ranges, buffer = new ArrayBuffer(12)) {
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
    await gather?.close().catch(() => {});
    threads.Worker = OriginalWorker; syncBuiltinESMExports();
    await Promise.allSettled(workers.map(worker => OriginalWorker.prototype.terminate.call(worker)));
    await fs.rm(f.root, { recursive: true, force: true });
  };
  try { gather = await createVerifiedPieceGather(f.options); S[I.enabled] = 1; }
  catch (error) { await cleanup(); throw error; }
  return { gather, workers, acks, outputs, cleanup,
    ranges: f.ranges.map((range, shard) => ({ shard, offset: range.offset, length: range.length })) };
}

test('three direct reads and bounded digests overlap, publishing one ordered authenticated output', { timeout: 5000 }, async () => {
  const f = await fixture(), h = await direct(f), S = f.state;
  try {
    h.send(); await until(() => S[I.reads] === 3);
    assert.equal(S[I.readDone], 0); assert.equal(S[I.maxReads], 3);
    Atomics.store(S, I.readGate + 2, 1); await until(() => S[I.hashes] === 1);
    assert.equal(S[I.readDone], 1); assert.equal(h.publications.length, 0);
    Atomics.store(S, I.releaseReads, 1); await until(() => S[I.hashes] === 3);
    assert.equal(S[I.hashDone], 0); assert.equal(S[I.maxHashes], 3);
    const active = await h.receipt();
    assert.equal(active.activeDigestCopyBytes, 12); assert.equal(active.peakDigestCopyBytes, 12);
    assert(active.activeDigestCopyBytes <= 2 * f.maxPieceBytes);
    Atomics.store(S, I.hashGate, 1); await until(() => S[I.hashDone] === 1);
    assert.equal(h.publications.length, 0, 'A fast digest cannot publish a partial batch');
    Atomics.store(S, I.releaseHashes, 1);
    const message = await h.take(value => value.buffer);
    assert.equal(message.error, undefined); assert.deepEqual([...new Uint8Array(message.buffer)], f.payloads.flatMap(bytes => [...bytes]));
    const receipt = await h.receipt();
    assert.equal(receipt.activeDigestCopyBytes, 0); assert.equal(receipt.directPieceReads, 3);
  } finally { await h.cleanup(); }
});

test('one rejected read waits for late sibling writes and hashes before zero, then recovers', { timeout: 5000 }, async () => {
  const f = await fixture(), h = await direct(f), S = f.state;
  try {
    Atomics.store(S, I.readError, 1); h.send();
    await until(() => S[I.reads] === 3 && S[I.readDone] === 1);
    assert.equal(h.publications.length, 0);
    Atomics.store(S, I.releaseReads, 1); await until(() => S[I.hashes] === 2);
    assert.equal(h.publications.length, 0, 'Read rejection must await outstanding hash owners');
    Atomics.store(S, I.releaseHashes, 1);
    const failed = await h.take(value => value.buffer);
    assert.match(failed.error, /positioned read failure/); assert(new Uint8Array(failed.buffer).every(byte => byte === 0));
    assert.equal(S[I.activeReads], 0); assert.equal(S[I.activeHashes], 0);
    assert.equal((await h.receipt()).activeDigestCopyBytes, 0);
    Atomics.store(S, I.readError, -1); h.send(f.ranges, failed.buffer);
    const recovered = await h.take(value => value.buffer);
    assert.deepEqual([...new Uint8Array(recovered.buffer)], f.payloads.flatMap(bytes => [...bytes]));
  } finally { await h.cleanup(); }
});

test('async SHA detects corrupt content and post-read inode swaps without publishing a prefix', { timeout: 5000 }, async () => {
  for (const change of ['corrupt', 'swap']) {
    const f = await fixture(), h = await direct(f), S = f.state, filename = f.files['down.bin'];
    try {
      if (change === 'corrupt') await fs.writeFile(filename, Uint8Array.of(0, 0, 0, 0));
      Atomics.store(S, I.releaseReads, 1); h.send(); await until(() => S[I.hashes] === 3);
      if (change === 'swap') { await fs.rename(filename, filename + '.saved'); await fs.writeFile(filename, f.payloads[2]); }
      Atomics.store(S, I.releaseHashes, 1);
      const failed = await h.take(value => value.buffer);
      assert.match(failed.error, change === 'corrupt' ? /integrity/ : /identity changed/);
      assert(new Uint8Array(failed.buffer).every(byte => byte === 0));
      assert.equal((await h.receipt()).activeDigestCopyBytes, 0);
      if (change === 'swap') { await fs.unlink(filename); await fs.rename(filename + '.saved', filename); }
      else await fs.writeFile(filename, f.payloads[2]);
      h.send(f.ranges, failed.buffer); const recovered = await h.take(value => value.buffer);
      assert.equal(recovered.error, undefined);
      assert.deepEqual([...new Uint8Array(recovered.buffer)], f.payloads.flatMap(bytes => [...bytes]));
    } finally { await h.cleanup(); }
  }
});

test('digest copy threshold keeps synchronous SHA and any partial range keeps serialized acquisition', { timeout: 5000 }, async () => {
  for (const partial of [false, true]) {
    const f = await fixture(4), h = await direct(f), S = f.state;
    try {
      if (partial) h.send([f.ranges[0], { ...f.ranges[1], offset: 1, length: 2 }]);
      else { Atomics.store(S, I.releaseReads, 1); h.send(); }
      // Generic authentication can use WebCrypto; allow it to finish normally.
      Atomics.store(S, I.releaseHashes, 1);
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

test('lane close rejects immediately but awaits slowest native/crypto job and closed descriptors before termination', { timeout: 10000 }, async () => {
  for (const phase of ['read', 'hash']) {
    const f = await fixture(), S = f.state, h = await publicHarness(f), { gather, acks, outputs } = h;
    try {
      if (phase === 'hash') Atomics.store(S, I.releaseReads, 1);
      const pending = gather.rangeLease.readMany(h.ranges, new AbortController().signal);
      await until(() => S[phase === 'read' ? I.reads : I.hashes] === 3);
      const rejected = assert.rejects(pending, error => {
        assert.equal(error.message, 'Gather reader closed.'); return true;
      });
      let settled = false; const closing = gather.close().then(() => { settled = true; });
      await rejected;
      assert.equal(settled, false); assert.equal(S[I.terminate], 0); assert.equal(acks.length, 0);
      const base = phase === 'read' ? I.readGate : I.hashGate;
      Atomics.store(S, base + 2, 1);
      await until(() => S[phase === 'read' ? I.readDone : I.hashDone] === 1);
      assert.equal(settled, false); assert.equal(acks.length, 0);
      Atomics.store(S, I.releaseReads, 1); Atomics.store(S, I.releaseHashes, 1);
      await closing;
      assert.equal(S[I.readDone], 3); assert.equal(S[I.activeReads], 0); assert.equal(S[I.activeHashes], 0);
      assert.equal(S[I.openedFDs], 3); assert.equal(S[I.closedFDs], 3);
      assert.equal(acks.length, 1); assert.equal(acks[0].receipt.activeDigestCopyBytes, 0);
      assert.equal(S[I.terminate], 1);
      assert(outputs.every(message => new Uint8Array(message.buffer).every(byte => byte === 0)));
    } finally { await h.cleanup(); }
  }
});

test('cancellation awaits owned jobs without publication and the same lane recovers', { timeout: 5000 }, async () => {
  const f = await fixture(), S = f.state, h = await publicHarness(f);
  try {
    Atomics.store(S, I.releaseReads, 1);
    const controller = new AbortController();
    const pending = h.gather.rangeLease.readMany(h.ranges, controller.signal);
    await until(() => S[I.hashes] === 3);
    controller.abort(new Error('Cancelled parallel batch'));
    const rejected = assert.rejects(pending, /Cancelled parallel batch/);
    Atomics.store(S, I.releaseHashes, 1); await rejected;
    assert.equal((await h.gather.getReceipt()).activeDigestCopyBytes, 0);
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

test('merged async copy evidence stays within the unchanged four-lane allowance', { timeout: 5000 }, async () => {
  const f = await fixture(), S = f.state;
  f.options.concurrency = 4; f.options.memoryBudgetBytes *= 4;
  const h = await publicHarness(f);
  try {
    Atomics.store(S, I.releaseReads, 1);
    const reads = Array.from({ length: 4 }, () => h.gather.rangeLease.readMany(h.ranges, new AbortController().signal));
    await until(() => S[I.hashes] === 12);
    Atomics.store(S, I.releaseHashes, 1);
    for (const lease of await Promise.all(reads)) lease.release();
    const receipt = await h.gather.getReceipt();
    assert.equal(receipt.activeDigestCopyBytes, 0); assert.equal(receipt.peakDigestCopyBytes, 48);
    assert(receipt.peakDigestCopyBytes <= 4 * 2 * f.maxPieceBytes);
    assert.equal(h.gather.rangeLease.memoryBytes, f.options.memoryBudgetBytes);
  } finally { await h.cleanup(); }
});
