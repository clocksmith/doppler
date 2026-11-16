import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, writeFile, rm, symlink, unlink } from 'node:fs/promises';
import { constants, openSync, closeSync, writeSync, readdirSync, fstatSync, statSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createRequire, syncBuiltinESMExports } from 'node:module';
import { createVerifiedPieceGather } from '../../src/storage/verified-piece-gather.js';

const sha = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const encode = value => new TextEncoder().encode(JSON.stringify(value));
const bytes = Uint8Array.of(1, 2, 3, 4);
const manifestBytes = encode({ shards: [{ filename: 'weights.bin', size: 4 }] });
const indexBytes = encode({ schema: 'doppler.verified-pieces/v1', manifestIdentity: sha(manifestBytes), maxPieceBytes: 4,
  files: [{ path: 'weights.bin', size: 4, pieces: [{ offset: 0, size: 4, identity: sha(bytes) }] }] });
const directory = await mkdtemp(join(tmpdir(), 'gather-ownership-'));
const filename = join(directory, 'weights.bin');
const options = { manifestBytes, indexBytes, indexIdentity: sha(indexBytes), files: { 'weights.bin': filename },
  maxPartBytes: 4, memoryBudgetBytes: 12 };
const threads = createRequire(import.meta.url)('node:worker_threads'), OriginalWorker = threads.Worker;
const workers = [];
let reader, releaseWriter;
try {
  await writeFile(filename, bytes);
  const sourceStat = statSync(filename);
  const countSourceHandles = () => readdirSync('/dev/fd').filter(name => {
    try { const stat = fstatSync(Number(name)); return stat.dev === sourceStat.dev && stat.ino === sourceStat.ino; }
    catch { return false; }
  }).length;
  let failures = 1, calls = 0;
  threads.Worker = class extends OriginalWorker {
    constructor(...args) { super(...args); workers.push(this); }
    terminate() {
      calls++;
      if (failures-- > 0) return Promise.reject(new Error('Injected termination failure before exit'));
      return super.terminate();
    }
  };
  syncBuiltinESMExports();
  reader = await createVerifiedPieceGather(options);
  const lease = await reader.readLease('weights.bin', 0, 4);
  assert.deepEqual([...lease.bytes], [...bytes]); lease.release();
  assert.equal(countSourceHandles(), 1, 'The worker owns exactly one lazy descriptor for this source');
  const first = reader.close();
  assert.equal(reader.close(), first, 'Concurrent close calls share the same retirement attempt');
  await assert.rejects(first, /termination failure/);
  assert(workers[0].threadId > 0, 'Injected failure precedes actual worker retirement');
  await assert.rejects(reader.readLease('weights.bin', 0, 4), /closed/);
  await reader.close();
  assert.equal(calls, 2, 'Failed retirement remains retryable without reopening admission');
  assert.equal(workers[0].threadId, -1);
  assert.equal(countSourceHandles(), 0, 'Actual worker retirement closes its synchronous file descriptor');

  calls = 0; failures = 1;
  const prior = workers.length;
  threads.Worker = class extends OriginalWorker {
    constructor(...args) { super(...args); workers.push(this); }
    terminate() {
      calls++;
      if (failures-- > 0) throw new Error('Injected synchronous termination failure');
      return super.terminate();
    }
  };
  syncBuiltinESMExports();
  reader = await createVerifiedPieceGather({ ...options, concurrency: 2, memoryBudgetBytes: 24 });
  const concurrent = await Promise.all([reader.readLease('weights.bin', 0, 4), reader.readLease('weights.bin', 0, 4)]);
  concurrent.forEach(lease => lease.release());
  assert.equal(countSourceHandles(), 2);
  await assert.rejects(reader.close(), /synchronous termination failure/);
  assert.equal(workers[prior + 1].threadId, -1, 'A synchronous close failure cannot skip another owned lane');
  assert.equal(countSourceHandles(), 1);
  await reader.close();
  assert.equal(calls, 3);
  assert.equal(countSourceHandles(), 0);

  threads.Worker = OriginalWorker; syncBuiltinESMExports();
  reader = await createVerifiedPieceGather(options);
  const retained = await reader.readLease('weights.bin', 0, 4);
  await reader.close();
  assert.deepEqual([...retained.bytes], [...bytes], 'Closing a pool cannot revoke a published consumer-owned lease');
  retained.release();
  assert.equal(retained.bytes.byteLength, 0, 'A late release detaches its own buffer after pool closure');
  assert.equal(countSourceHandles(), 0);
  reader = await createVerifiedPieceGather(options);
  await unlink(filename);
  await symlink(join(directory, 'other.bin'), filename);
  await writeFile(join(directory, 'other.bin'), bytes);
  await assert.rejects(reader.readLease('weights.bin', 0, 4), /symbolic|symlink|ELOOP/i,
    'A leaf replaced with a symlink cannot acquire bytes outside the approved regular filename');
  await reader.close();
  await unlink(filename);
  execFileSync('mkfifo', [filename]);
  reader = await createVerifiedPieceGather(options);
  // On the old path this releases a blocking FIFO open, so a failing regression
  // cannot strand the test worker. The corrected path rejects before this timer.
  releaseWriter = setTimeout(() => {
    let handle;
    try { handle = openSync(filename, constants.O_WRONLY | constants.O_NONBLOCK); writeSync(handle, bytes); }
    catch (error) { if (error.code !== 'ENXIO') throw error; }
    finally { if (handle !== undefined) closeSync(handle); }
  }, 200);
  const outcome = await reader.readLease('weights.bin', 0, 4).then(lease => {
    lease.release(); return null;
  }, error => error);
  assert.match(outcome?.message ?? '', /regular file/, 'Non-regular acquisition must fail before a blocking read');
} finally {
  clearTimeout(releaseWriter);
  threads.Worker = OriginalWorker; syncBuiltinESMExports();
  await reader?.close().catch(() => {});
  await Promise.allSettled(workers.map(worker => OriginalWorker.prototype.terminate.call(worker)));
  await rm(directory, { recursive: true, force: true });
}
console.log('verified gather ownership: retryable retirement, descriptor release, regular files and no leaf symlink acquisition');
