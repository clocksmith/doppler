import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createRequire, syncBuiltinESMExports } from 'node:module';
import { createVerifiedPieceStorage } from '../../src/storage/verified-piece-storage.js';
import { createVerifiedPieceGather } from '../../src/storage/verified-piece-gather.js';

// Fault injection observes real worker creation. No production test hook or model is needed.
const threads = createRequire(import.meta.url)('node:worker_threads');
const OriginalWorker = threads.Worker;
const sha = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const encode = value => new TextEncoder().encode(JSON.stringify(value));
const bytes = Uint8Array.of(1, 2, 3, 4);
const manifestBytes = encode({ shards: [{ filename: 'weights.bin', size: 4 }] });
const indexBytes = encode({ schema: 'doppler.verified-pieces/v1', manifestIdentity: sha(manifestBytes), maxPieceBytes: 4,
  files: [{ path: 'weights.bin', size: 4, pieces: [{ offset: 0, size: 4, identity: sha(bytes) }] }] });
const directory = await mkdtemp(join(tmpdir(), 'gather-startup-'));
let controller = new AbortController(), abortAt = 2, reportCleanupFailure = false;
const workers = [], exits = [];
let opened;
try {
  await writeFile(join(directory, 'weights.bin'), bytes);
  threads.Worker = class extends OriginalWorker {
    constructor(...args) {
      super(...args);
      if (String(args[0]).endsWith('/verified-piece-gather-worker.js')) {
        workers.push(this);
        exits.push(new Promise(resolve => this.once('exit', resolve)));
        if (workers.length === abortAt) controller.abort(new Error('Cancel during worker startup'));
      }
    }
    async terminate() {
      const code = await super.terminate();
      if (reportCleanupFailure) { reportCleanupFailure = false; throw new Error('Injected worker cleanup failure'); }
      return code;
    }
  };
  syncBuiltinESMExports();
  let failure;
  try {
    opened = await createVerifiedPieceStorage({ manifestBytes, indexBytes, indexIdentity: sha(indexBytes),
      signal: controller.signal, acquire: async () => bytes,
      nodeGather: { files: { 'weights.bin': join(directory, 'weights.bin') },
        maxPartBytes: 4, memoryBudgetBytes: 48, concurrency: 4 } });
  } catch (error) { failure = error; }
  if (opened) await opened.storage.close();
  assert.equal(opened, undefined, 'A cancelled source is never published');
  assert.match(failure?.message ?? '', /Cancel during worker startup/);
  // Lanes start together, so a cancellation during startup finds all four spawned; none may survive it.
  assert.equal(workers.length, 4, 'Lanes start together rather than one after another');
  assert(workers.every(worker => worker.threadId === -1), 'All partial owners exited before factory rejection settled');
  await Promise.all(exits);
  const before = workers.length;
  await assert.rejects(createVerifiedPieceGather({ manifestBytes, indexBytes, indexIdentity: sha(indexBytes),
    files: { 'weights.bin': join(directory, 'weights.bin') }, maxPartBytes: 4, memoryBudgetBytes: 12,
    startupSignal: AbortSignal.abort(new Error('Cancel before startup')) }), /Cancel before startup/);
  assert.equal(workers.length, before, 'An already cancelled admission never spawns a worker');

  controller = new AbortController(); abortAt = workers.length + 2; reportCleanupFailure = true;
  await assert.rejects(createVerifiedPieceStorage({ manifestBytes, indexBytes, indexIdentity: sha(indexBytes),
    signal: controller.signal, acquire: async () => bytes,
    nodeGather: { files: { 'weights.bin': join(directory, 'weights.bin') },
      maxPartBytes: 4, memoryBudgetBytes: 48, concurrency: 4 } }), error => {
    assert(error instanceof AggregateError, 'Startup cancellation cannot hide cleanup failure');
    assert.match(String(error.errors[0]), /Cancel during worker startup/);
    assert.match(String(error.errors[1]), /Injected worker cleanup failure/);
    return true;
  });
  assert(workers.every(worker => worker.threadId === -1), 'One cleanup failure cannot skip another partial owner');

  abortAt = Infinity;
  opened = await createVerifiedPieceStorage({ manifestBytes, indexBytes, indexIdentity: sha(indexBytes),
    acquire: async () => bytes, nodeGather: { files: { 'weights.bin': join(directory, 'weights.bin') },
      maxPartBytes: 4, memoryBudgetBytes: 24, concurrency: 2 } });
  const recovered = await opened.storage.rangeLease.read(0, 0, 4, new AbortController().signal);
  assert.deepEqual([...recovered.bytes], [...bytes]); recovered.release();
  await opened.storage.close();
} finally {
  threads.Worker = OriginalWorker; syncBuiltinESMExports();
  await opened?.storage.close();
  await Promise.allSettled(workers.map(worker => worker.terminate()));
  await rm(directory, { recursive: true });
}
console.log('verified gather startup: no cancelled publication, settled partial workers, visible cleanup failure and new-source recovery');
