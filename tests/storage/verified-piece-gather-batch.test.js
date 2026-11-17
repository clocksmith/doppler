import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createHash } from 'node:crypto';
import { Worker } from 'node:worker_threads';
import fs from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createVerifiedPieceGather } from '../../src/storage/verified-piece-gather.js';

const sha = bytes => 'sha256:' + createHash('sha256').update(bytes).digest('hex');
const encode = value => new TextEncoder().encode(JSON.stringify(value));

test('one bounded batch authenticates all ranges before one transfer and owns every view until release', { timeout: 10000 }, async () => {
  const root = await fs.mkdtemp(join(tmpdir(), 'gather-batch-'));
  const payloads = [Uint8Array.of(1, 2, 3, 4, 5, 6, 7, 8), Uint8Array.of(9, 10, 11, 12, 13, 14, 15, 16)];
  const names = ['gate.bin', 'down.bin'];
  const files = Object.fromEntries(names.map(name => [name, join(root, name)]));
  const manifestBytes = encode({ shards: names.map(filename => ({ filename, size: 8 })) });
  const indexBytes = encode({ schema: 'doppler.verified-pieces/v1', manifestIdentity: sha(manifestBytes), maxPieceBytes: 4,
    files: names.map((path, index) => ({ path, size: 8, pieces: [0, 4].map(offset => ({ offset, size: 4,
      identity: sha(payloads[index].subarray(offset, offset + 4)) })) })) });
  const posted = [], failedOutputs = [];
  const post = Worker.prototype.postMessage, emit = Worker.prototype.emit;
  Worker.prototype.postMessage = function(message, ...args) {
    if (message.ranges) posted.push({ ranges: structuredClone(message.ranges), length: message.length,
      allocation: message.buffer.byteLength, transfers: args[0]?.length });
    return post.call(this, message, ...args);
  };
  Worker.prototype.emit = function(event, ...args) {
    if (event === 'message' && args[0]?.error && args[0].buffer) failedOutputs.push([...new Uint8Array(args[0].buffer)]);
    return emit.call(this, event, ...args);
  };
  let reader;
  try {
    await Promise.all(names.map((name, index) => fs.writeFile(files[name], payloads[index])));
    const identities = Object.fromEntries(await Promise.all(names.map(async name => {
      const stat = await fs.stat(files[name], { bigint: true });
      return [name, { device: stat.dev.toString(), inode: stat.ino.toString(), size: stat.size.toString() }];
    })));
    const options = { manifestBytes, indexBytes, indexIdentity: sha(indexBytes), files, fileIdentities: identities,
      maxPartBytes: 12, concurrency: 2, memoryBudgetBytes: 40 };
    await assert.rejects(createVerifiedPieceGather({ ...options, memoryBudgetBytes: 39 }), /budget/);
    reader = await createVerifiedPieceGather(options);
    assert.equal(reader.rangeLease.memoryBytes, 2 * (12 + 2 * 4));
    const ranges = [{ shard: 0, offset: 0, length: 4 }, { shard: 0, offset: 4, length: 4 }, { shard: 1, offset: 0, length: 4 }];
    const opening = reader.rangeLease.readMany(ranges, new AbortController().signal);
    ranges[0].offset = 4; ranges.pop();
    const lease = await opening;
    assert.deepEqual([...lease.bytes], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    assert.equal(posted.length, 1, 'Three ranges use one worker request');
    assert.equal(posted[0].allocation, 12, 'One admitted output, not one output per range');
    assert.equal(posted[0].transfers, 1);
    const second = await reader.rangeLease.readMany([{ shard: 1, offset: 2, length: 5 }], new AbortController().signal);
    assert.deepEqual([...second.bytes], [11, 12, 13, 14, 15]);
    await assert.rejects(reader.rangeLease.readMany([{ shard: 0, offset: 0, length: 4 }], new AbortController().signal), /Release/);
    await assert.rejects(reader.getReceipt(), /Release/);
    const gateView = lease.bytes.subarray(0, 4), downView = lease.bytes.subarray(8);
    const beforeRelease = posted.length;
    lease.release(); lease.release();
    assert.equal(lease.bytes.byteLength, 0); assert.equal(gateView.byteLength, 0); assert.equal(downView.byteLength, 0);
    assert.equal(second.bytes.byteLength, 5, 'Sibling lane remains owned');
    assert.equal(posted.length, beforeRelease, 'Release is local recycling, not worker IPC');
    second.release();
    const receipt = await reader.getReceipt();
    assert.equal(receipt.batchReads, 2); assert.equal(receipt.batchRanges, 4); assert.equal(receipt.batchBytes, 17);
    assert.equal(receipt.directPieceReads, 3); assert.equal(receipt.directPieceBytes, 12);
    const readCount = posted.length;
    for (const invalid of [[], Array(4).fill({ shard: 0, offset: 0, length: 1 }),
      [{ shard: 0, offset: 0, length: 0 }], [{ shard: 0, offset: 0, length: 13 }],
      [{ shard: 0, offset: 0, length: 8 }, { shard: 1, offset: 0, length: 8 }],
      [{ shard: 2, offset: 0, length: 1 }], [{ shard: 0, offset: -1, length: 1 }]]) {
      await assert.rejects(reader.rangeLease.readMany(invalid, new AbortController().signal), /bound|ranges|shard/);
    }
    assert.equal(posted.length, readCount, 'Invalid batches never transfer an output');
    await fs.writeFile(files['down.bin'], Uint8Array.of(0, 0, 0, 0, 13, 14, 15, 16));
    await assert.rejects(reader.rangeLease.readMany([{ shard: 0, offset: 0, length: 4 },
      { shard: 1, offset: 0, length: 4 }], new AbortController().signal), /integrity/);
    assert(failedOutputs.at(-1).every(byte => byte === 0), 'A later hash failure zeros the entire output allocation');
    await fs.writeFile(files['down.bin'], payloads[1]);
    const controller = new AbortController();
    const pending = reader.rangeLease.readMany([{ shard: 0, offset: 0, length: 4 }, { shard: 1, offset: 0, length: 4 }], controller.signal);
    controller.abort(new Error('Cancelled expert batch'));
    await assert.rejects(pending, /Cancelled expert batch/);
    const recovered = await reader.rangeLease.readMany([{ shard: 1, offset: 0, length: 4 }], new AbortController().signal);
    assert.deepEqual([...recovered.bytes], [9, 10, 11, 12]); recovered.release();
    const preaborted = new AbortController(); preaborted.abort(new Error('Pre-aborted batch'));
    const beforeAbort = posted.length;
    await assert.rejects(reader.rangeLease.readMany([{ shard: 0, offset: 0, length: 4 }], preaborted.signal), /Pre-aborted/);
    assert.equal(posted.length, beforeAbort);
    await fs.rename(files['down.bin'], files['down.bin'] + '.retained');
    await fs.writeFile(files['down.bin'], payloads[1]);
    await assert.rejects(reader.rangeLease.readMany([{ shard: 0, offset: 0, length: 4 },
      { shard: 1, offset: 0, length: 4 }], new AbortController().signal), /identity changed/);
    await fs.unlink(files['down.bin']); await fs.rename(files['down.bin'] + '.retained', files['down.bin']);
    const retained = await reader.rangeLease.readMany([{ shard: 0, offset: 0, length: 4 }], new AbortController().signal);
    const closingRead = reader.rangeLease.readMany([{ shard: 0, offset: 0, length: 4 },
      { shard: 1, offset: 0, length: 4 }], new AbortController().signal);
    const rejectedClose = assert.rejects(closingRead, /closed/);
    await reader.close(); await rejectedClose;
    assert.equal(retained.bytes.byteLength, 4, 'Published lease survives owner close until explicit release');
    retained.release(); assert.equal(retained.bytes.byteLength, 0);
    await assert.rejects(reader.rangeLease.readMany([{ shard: 0, offset: 0, length: 4 }], new AbortController().signal), /closed/);
  } finally {
    await reader?.close(); Worker.prototype.postMessage = post; Worker.prototype.emit = emit;
    await fs.rm(root, { recursive: true, force: true });
  }
});
