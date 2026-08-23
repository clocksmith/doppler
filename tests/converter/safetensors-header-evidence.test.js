import assert from 'node:assert/strict';
import { sha256BytesHex } from '../../src/utils/sha256.js';
import {
  materializeSafetensorsHeaderEvidence,
  parseSafetensorsHeaderEvidence,
  readSafetensorsHeaderLength,
} from '../../src/converter/safetensors-header-evidence.js';

function headerBytes(header) {
  const payload = new TextEncoder().encode(JSON.stringify(header));
  const bytes = new Uint8Array(payload.byteLength + 8);
  new DataView(bytes.buffer).setBigUint64(0, BigInt(payload.byteLength), true);
  bytes.set(payload, 8);
  return bytes;
}

const shardA = headerBytes({
  __metadata__: { format: 'pt' },
  'model.embed_tokens.weight': { dtype: 'BF16', shape: [16, 8], data_offsets: [0, 256] },
});
const shardB = headerBytes({
  'lm_head.weight': { dtype: 'BF16', shape: [16, 8], data_offsets: [0, 256] },
});

assert.equal(readSafetensorsHeaderLength(shardA.subarray(0, 8)), shardA.byteLength - 8);
assert.deepEqual(
  parseSafetensorsHeaderEvidence(shardA, {
    sourceFile: 'a.safetensors',
    expectedSha256: sha256BytesHex(shardA),
  }).tensors['model.embed_tokens.weight'],
  { dtype: 'BF16', shape: [16, 8], sourceFile: 'a.safetensors' }
);

const byFile = new Map([
  ['a.safetensors', shardA],
  ['b.safetensors', shardB],
]);
const pin = {
  schema: 'doppler.safetensors-header-pin/v1',
  checkpointId: 'example/model',
  repository: 'example/model',
  revision: '0123456789abcdef',
  shards: [
    { sourceFile: 'a.safetensors', headerSha256: sha256BytesHex(shardA) },
    { sourceFile: 'b.safetensors', headerSha256: `sha256:${sha256BytesHex(shardB)}` },
  ],
};
const receipt = await materializeSafetensorsHeaderEvidence(pin, async ({ sourceFile, start, end }) => (
  byFile.get(sourceFile).subarray(start, end + 1)
));
assert.equal(receipt.tensorCount, 2);
assert.deepEqual(Object.keys(receipt.tensors), ['lm_head.weight', 'model.embed_tokens.weight']);
assert.equal(receipt.additionalSourceHeaders[0].tensorCount, 1);

assert.throws(
  () => parseSafetensorsHeaderEvidence(shardA, {
    sourceFile: 'a.safetensors',
    expectedSha256: '0'.repeat(64),
  }),
  /digest mismatch/
);

const duplicatePin = {
  ...pin,
  shards: [
    { sourceFile: 'a.safetensors', headerSha256: sha256BytesHex(shardA) },
    { sourceFile: 'a-copy.safetensors', headerSha256: sha256BytesHex(shardA) },
  ],
};
await assert.rejects(
  () => materializeSafetensorsHeaderEvidence(duplicatePin, async ({ sourceFile, start, end }) => (
    (sourceFile === 'a.safetensors' ? shardA : shardA).subarray(start, end + 1)
  )),
  /multiple SafeTensors shards/
);

console.log('✔ safetensors-header-evidence.test.js passed');
