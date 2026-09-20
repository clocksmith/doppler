import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { createNodeCapsuleArtifactStore } from '../../src/tooling/node-capsule-artifact-store.js';
import { createVerifiedCapsuleArtifactStore } from '../../src/client/runtime/verified-capsule-artifact-store.js';

const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-capsule-stream-'));
try {
  const bytes = new Uint8Array(1024 * 1024 + 17).fill(71);
  await fs.writeFile(path.join(directory, 'weights.bin'), bytes);
  const artifact = { artifactId: 'weights', path: 'weights.bin', role: 'weight-shard', sizeBytes: bytes.length,
    hash: `sha256:${createHash('sha256').update(bytes).digest('hex')}` };
  const source = createNodeCapsuleArtifactStore(path.join(directory, 'capsule.json'));
  const store = createVerifiedCapsuleArtifactStore({ artifacts: [artifact] }, source, { maxAcquisitionChunkBytes: 17003 });
  await store.hashArtifact(artifact);
  assert.equal(store.getMetrics().peakSourceChunkBytes, 17003);
  assert.equal(store.getMetrics().streamedSourceBytes, bytes.length);
  assert.deepEqual(await store.readArtifactRange(artifact, 65533, 19), bytes.subarray(65533, 65552));
  store.close();
  const controller = new AbortController();
  const stream = source.streamArtifact(artifact, { signal: controller.signal, maxChunkBytes: 4096 });
  assert.equal((await stream.next()).value.length, 4096);
  controller.abort(new Error('stop file acquisition'));
  await assert.rejects(stream.next(), /stop file acquisition/);
  for (const delta of [-1, 1]) {
    await assert.rejects(async () => {
      for await (const chunk of source.streamArtifact({ ...artifact, sizeBytes: bytes.length + delta }, { maxChunkBytes: 17003 })) void chunk;
    }, /size mismatch|byte limit/);
  }
} finally { await fs.rm(directory, { recursive: true, force: true }); }
console.log('capsule-file-stream: passed');
