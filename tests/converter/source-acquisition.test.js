import assert from 'node:assert/strict';
import { sha256BytesHex } from '../../src/utils/sha256.js';
import { createSourceAcquisitionReceipt } from '../../src/converter/source-acquisition.js';

const files = new Map([
  ['config.json', new TextEncoder().encode('{"model_type":"example"}')],
  ['model.safetensors', new Uint8Array([1, 2, 3, 4])],
]);
const policy = {
  schema: 'doppler.source-acquisition/v1',
  checkpointId: 'example/model',
  repository: 'example/model',
  revision: '0123456789abcdef',
  files: [...files].map(([file, bytes]) => ({
    path: file,
    size: bytes.byteLength,
    sha256: sha256BytesHex(bytes),
    role: file.endsWith('.json') ? 'model-config' : 'weight-shard',
  })),
  author: { kind: 'tool', actor: 'source-acquisition-test' },
};
const adapters = {
  async listFiles() { return [...files.keys()]; },
  async statFile(file) { return files.get(file).byteLength; },
  async hashFile(file) { return sha256BytesHex(files.get(file)); },
};
const receipt = await createSourceAcquisitionReceipt(policy, adapters);
assert.equal(receipt.complete, true);
assert.equal(receipt.fileCount, 2);
assert.equal(receipt.totalBytes, 28);
assert.ok(receipt.files.every((file) => file.verified));

await assert.rejects(
  () => createSourceAcquisitionReceipt(policy, { ...adapters, listFiles: async () => ['config.json'] }),
  /1 missing/
);
await assert.rejects(
  () => createSourceAcquisitionReceipt(policy, { ...adapters, statFile: async () => 1 }),
  /size mismatch/
);
await assert.rejects(
  () => createSourceAcquisitionReceipt(policy, { ...adapters, hashFile: async () => '0'.repeat(64) }),
  /digest mismatch/
);

console.log('✔ source-acquisition.test.js passed');
