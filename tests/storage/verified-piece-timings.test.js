import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Worker } from 'node:worker_threads';
import { createHash } from 'node:crypto';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { snapshotVerifiedPieceJobTimings, snapshotVerifiedRangeLeaseTimings } from '../../src/storage/verified-piece-timings.js';
import { createVerifiedPieceGather } from '../../src/storage/verified-piece-gather.js';

const job = { workerWallMs: 4, readWaitMs: 5, verificationMs: 1, copyMs: 0 };
const full = { ...job, roundTripMs: 8 };
test('bounded timing snapshots reject malformed present metadata and preserve explicit unsupported omission', () => {
  assert.equal(snapshotVerifiedPieceJobTimings(undefined), undefined);
  assert.equal(snapshotVerifiedRangeLeaseTimings(undefined), undefined);
  const snapshot = snapshotVerifiedRangeLeaseTimings(full);
  assert(Object.isFrozen(snapshot)); assert.deepEqual(snapshot, full); assert.notEqual(snapshot, full);
  for (const bad of [null, [], 0, {}, job, { ...full, extra: 0 }, { ...full, copyMs: -1 },
    { ...full, workerWallMs: Infinity }, { ...full, verificationMs: NaN },
    { ...full, roundTripMs: '1' }, { ...full, readWaitMs: Number.MAX_SAFE_INTEGER + 1 }]) {
    assert.throws(() => snapshotVerifiedRangeLeaseTimings(bad), /timing metadata/);
  }
  let accessed = false;
  const getter = { ...full }; Object.defineProperty(getter, 'copyMs', { enumerable: true, get() { accessed = true; return 0; } });
  assert.throws(() => snapshotVerifiedRangeLeaseTimings(getter), /timing metadata/); assert.equal(accessed, false);
});

test('real lane rejects bad timing replies without losing its buffer, accepts legacy omission and recovers', { timeout: 5000 }, async () => {
  const root = await mkdtemp(join(tmpdir(), 'gather-timings-')), filename = join(root, 'weights.bin');
  const bytes = Uint8Array.of(1, 2, 3, 4), encode = value => new TextEncoder().encode(JSON.stringify(value));
  const sha = value => 'sha256:' + createHash('sha256').update(value).digest('hex');
  const manifestBytes = encode({ shards: [{ filename: 'weights.bin', size: 4 }] });
  const indexBytes = encode({ schema: 'doppler.verified-pieces/v1', manifestIdentity: sha(manifestBytes), maxPieceBytes: 4,
    files: [{ path: 'weights.bin', size: 4, pieces: [{ offset: 0, size: 4, identity: sha(bytes) }] }] });
  const emit = Worker.prototype.emit, replies = []; let mode = 'malformed', reader;
  Worker.prototype.emit = function(event, ...args) {
    const message = args[0];
    if (event === 'message' && message?.buffer) {
      replies.push(message);
      if (!message.error && mode === 'malformed') { message.timings = { ...message.timings, readWaitMs: NaN }; mode = 'normal'; }
      else if (!message.error && mode === 'legacy') { delete message.timings; mode = 'normal'; }
    }
    return emit.call(this, event, ...args);
  };
  try {
    await writeFile(filename, bytes);
    reader = await createVerifiedPieceGather({ manifestBytes, indexBytes, indexIdentity: sha(indexBytes),
      files: { 'weights.bin': filename }, maxPartBytes: 4, memoryBudgetBytes: 12 });
    await assert.rejects(reader.readLease('weights.bin', 0, 4), /timing metadata/);
    assert(new Uint8Array(replies.at(-1).buffer).every(byte => byte === 0));
    mode = 'legacy'; const legacy = await reader.readLease('weights.bin', 0, 4);
    assert.equal(legacy.timings, undefined); legacy.release();
    const recovered = await reader.readLease('weights.bin', 0, 4);
    assert.deepEqual([...recovered.bytes], [...bytes]); assert(Object.isFrozen(recovered.timings));
    const timing = recovered.timings; recovered.release();
    assert.equal(recovered.bytes.byteLength, 0); assert.equal(recovered.timings, timing);
    await writeFile(filename, new Uint8Array(4));
    await assert.rejects(reader.readLease('weights.bin', 0, 4), /integrity/);
    assert.equal(replies.at(-1).timings, undefined, 'Worker failures cannot claim successful per-job timings');
  } finally { await reader?.close(); Worker.prototype.emit = emit; await rm(root, { recursive: true, force: true }); }
});
