import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { addHashesToFileEntries, MAX_NODE_READ_BYTES } from '../../src/tooling/source-runtime/node-loader.js';

const dir = await mkdtemp(path.join(tmpdir(), 'doppler-source-hash-'));
try {
  const file = path.join(dir, 'weights.bin');
  const bytes = Uint8Array.from({ length: 8197 }, (_, i) => i % 251);
  await writeFile(file, bytes);
  const expected = createHash('sha256').update(bytes).digest('hex');
  for (const chunkBytes of [257, 4096, 4 * 1024 * 1024, MAX_NODE_READ_BYTES + 1]) {
    const [entry] = await addHashesToFileEntries([{ path: file }], 'sha256', { readChunkBytes: chunkBytes, maxInFlightBytes: MAX_NODE_READ_BYTES });
    assert.equal(entry.hash, expected, 'chunking must not change the content identity');
    assert.equal(entry.size, bytes.length);
  }
  for (const invalid of [undefined, null, 0, -1, 1.5]) {
    await assert.rejects(addHashesToFileEntries([{ path: file }], 'sha256', { readChunkBytes: invalid, maxInFlightBytes: 4096 }), /chunkBytes/);
  }
  await assert.rejects(addHashesToFileEntries([{ path: file }], 'sha256', undefined), /chunkBytes/);
  await assert.rejects(addHashesToFileEntries([{ path: file }], 'sha256', { readChunkBytes: 4096, maxInFlightBytes: 0 }), /maxInFlightBytes/);
  await assert.rejects(addHashesToFileEntries([{ path: path.join(dir, 'missing') }], 'sha256', { readChunkBytes: 8192, maxInFlightBytes: 4096 }), /does not exist/);
  bytes[0] ^= 1;
  await writeFile(file, bytes);
  const [changed] = await addHashesToFileEntries([{ path: file }], 'sha256', { readChunkBytes: 8192, maxInFlightBytes: 4096 });
  assert.notEqual(changed.hash, expected, 'modified content must always be rehashed');
} finally {
  await rm(dir, { recursive: true, force: true });
}
console.log('Source hashing preserves byte identity with explicit bounded read chunks.');
