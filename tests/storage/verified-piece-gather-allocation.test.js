import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createHash } from 'node:crypto';
import { createRequire, syncBuiltinESMExports } from 'node:module';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createVerifiedPieceGather } from '../../src/storage/verified-piece-gather.js';

// Inject allocation failure, never a huge real allocation or a model/GPU load.
test('output allocation failure precedes worker ownership', { timeout: 5000 }, async () => {
  const threads = createRequire(import.meta.url)('node:worker_threads');
  const OriginalWorker = threads.Worker, OriginalArrayBuffer = globalThis.ArrayBuffer;
  const workers = [], bytes = Uint8Array.of(1, 2, 3, 4);
  const encode = value => new TextEncoder().encode(JSON.stringify(value));
  const sha = value => 'sha256:' + createHash('sha256').update(value).digest('hex');
  const manifestBytes = encode({ shards: [{ filename: 'weights.bin', size: 4 }] });
  const indexBytes = encode({ schema: 'doppler.verified-pieces/v1', manifestIdentity: sha(manifestBytes), maxPieceBytes: 4,
    files: [{ path: 'weights.bin', size: 4, pieces: [{ offset: 0, size: 4, identity: sha(bytes) }] }] });
  const directory = await mkdtemp(join(tmpdir(), 'gather-allocation-'));
  let fail = true, reader;
  try {
    await writeFile(join(directory, 'weights.bin'), bytes);
    threads.Worker = class extends OriginalWorker {
      constructor(...args) { super(...args); workers.push(this); }
    };
    syncBuiltinESMExports();
    globalThis.ArrayBuffer = class extends OriginalArrayBuffer {
      constructor(length) {
        if (length === 12 && fail) { fail = false; throw new RangeError('Injected output allocation failure'); }
        super(length);
      }
    };
    const options = { manifestBytes, indexBytes, indexIdentity: sha(indexBytes),
      files: { 'weights.bin': join(directory, 'weights.bin') }, maxPartBytes: 12, memoryBudgetBytes: 20 };
    await assert.rejects(createVerifiedPieceGather(options), /Injected output allocation failure/);
    assert.equal(workers.length, 0, 'No worker may outlive a failed output allocation');
    globalThis.ArrayBuffer = OriginalArrayBuffer;
    reader = await createVerifiedPieceGather(options);
    const lease = await reader.readLease('weights.bin', 0, 4);
    assert.deepEqual([...lease.bytes], [...bytes]); lease.release();
    await reader.close();
    assert(workers.every(worker => worker.threadId === -1));
  } finally {
    globalThis.ArrayBuffer = OriginalArrayBuffer;
    threads.Worker = OriginalWorker; syncBuiltinESMExports();
    await reader?.close();
    await Promise.allSettled(workers.map(worker => OriginalWorker.prototype.terminate.call(worker)));
    await rm(directory, { recursive: true, force: true });
  }
});
