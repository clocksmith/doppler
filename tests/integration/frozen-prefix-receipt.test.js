import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile, stat, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { frozenPrefixJsonChunks, writeFrozenPrefixReceipt, finishFrozenPrefixReplay } from '../helpers/frozen-prefix-receipt.js';

const directory = await mkdtemp(join(tmpdir(), 'doppler-prefix-receipt-'));
try {
  const receipt = { escaped: '"\\\n\ud800', absent: undefined, nil: null,
    observation: { timeline: Array.from({ length: 2000 }, (_, step) => ({ step,
      values: Array.from({ length: 1200 }, (_, i) => i / 3), missing: [undefined, null] })) } };
  const chunks = [...frozenPrefixJsonChunks(receipt)];
  assert(chunks.every(chunk => chunk.length < 30000));
  assert.equal(chunks.join(''), JSON.stringify(receipt));
  const destination = join(directory, 'receipt.json');
  await writeFrozenPrefixReceipt(destination, receipt);
  assert.deepEqual(JSON.parse(await readFile(destination, 'utf8')), JSON.parse(JSON.stringify(receipt)));
  await assert.rejects(stat(destination + '.partial'), { code: 'ENOENT' });
  const committedBytes = await readFile(destination, 'utf8');
  await assert.rejects(writeFrozenPrefixReceipt(destination, { unsupported: 1n }), TypeError);
  assert.equal(await readFile(destination, 'utf8'), committedBytes);

  const owned = join(directory, 'owned');
  await writeFile(owned, 'temporary');
  const calls = [];
  await assert.rejects(finishFrozenPrefixReplay({ destination, receipt, directory: owned,
    writeReceipt: async () => { throw new Error('injected write failure'); },
    browser: { close: async () => { calls.push('browser'); throw new Error('injected browser failure'); } },
    server: { closeAllConnections: () => calls.push('connections'), close: done => { calls.push('server'); done(); } },
  }), error => error instanceof AggregateError && error.errors.length === 2);
  assert.deepEqual(calls, ['browser', 'connections', 'server']);
  await assert.rejects(stat(owned), { code: 'ENOENT' });
  console.log('Frozen prefix chunking, atomic receipt, and failure cleanup passed.');
} finally { await rm(directory, { recursive: true, force: true }); }
