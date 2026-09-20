// Diagnostic fixture, not a performance benchmark or production acceptance gate.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createArtifactStorageContext } from '../../src/storage/artifact-storage-context.js';

const bytes = new Uint8Array(8 * 1024 * 1024).fill(23);
const digest = value => createHash('sha256').update(value).digest('hex');
const calls = [];
const context = createArtifactStorageContext({
  manifest: {
    modelId: 'diagnostic-owned-ranges',
    hashAlgorithm: 'sha256',
    shards: [{ filename: 'weights.bin', size: bytes.length, hash: digest(bytes) }],
  },
  expectedFormat: 'rdrr',
  verifyHashes: true,
  async readRange(path, offset, length) {
    calls.push({ path, offset, length,
      caller: new Error().stack.split('\n').slice(2, 3)[0].trim() });
    return bytes.slice(offset, offset + length).buffer;
  },
});
const loaded = await context.loadShard(0);
assert.equal(digest(new Uint8Array(loaded)), digest(bytes));
console.log(JSON.stringify({
  scope: 'Bounded synthetic range-copy diagnostic; no model or timing claim.',
  inputBytes: bytes.length,
  loadedBytes: loaded.byteLength,
  returnedBytes: calls.reduce((total, call) => total + call.length, 0),
  calls,
}, null, 2));
