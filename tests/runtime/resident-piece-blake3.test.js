import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { computeHash } from '../../src/storage/shard-manager.js';
import { createVerifiedCapsuleArtifactStore } from '../../src/client/runtime/verified-capsule-artifact-store.js';

const bytes = Uint8Array.of(1, 2, 3, 4);
const sha = value => 'sha256:' + createHash('sha256').update(value).digest('hex');
const encode = value => new TextEncoder().encode(JSON.stringify(value));
async function fixture(algorithm, change = () => {}) {
  const manifest = { modelId: 'piece-control', hashAlgorithm: algorithm, shards: [
    { filename: 'weights.bin', size: bytes.length, hash: await computeHash(bytes, algorithm) }] };
  const index = { schema: 'doppler.verified-pieces/v1', manifestIdentity: sha(encode(manifest)), maxPieceBytes: 2,
    files: [{ path: 'weights.bin', size: bytes.length, pieces: [0, 2].map(offset => (
      { offset, size: 2, identity: sha(bytes.subarray(offset, offset + 2)) })) }] };
  change(manifest, index);
  const source = new Map([['manifest', encode(manifest)], ['pieces', encode(index)], ['weights', bytes]]);
  const capsule = { modelId: manifest.modelId, program: { manifestArtifactId: 'manifest',
    pieceIndexArtifactId: 'pieces', weightArtifactIds: ['weights'] }, artifacts: [
    ['manifest', 'manifest', 'manifest.json'], ['pieces', 'piece-index', 'pieces.json'], ['weights', 'weight-shard', 'weights.bin']
  ].map(([artifactId, role, path]) => ({ artifactId, role, path,
    hash: sha(source.get(artifactId)), sizeBytes: source.get(artifactId).length })) };
  let corrupt = false;
  const reads = [], ranges = [];
  const store = createVerifiedCapsuleArtifactStore(capsule, {
    async readArtifact(artifact) { reads.push(artifact.artifactId); return source.get(artifact.artifactId); },
    async readArtifactRange(artifact, offset, length) {
      ranges.push({ artifactId: artifact.artifactId, offset, length });
      return corrupt ? new Uint8Array(length) : source.get(artifact.artifactId).slice(offset, offset + length);
    },
  });
  return { capsule, store, reads, ranges, corrupt: () => { corrupt = true; } };
}
for (const algorithm of ['sha256', 'blake3']) {
  const f = await fixture(algorithm);
  try {
    await f.store.enableResidentPieces();
    const weight = f.capsule.artifacts.find(row => row.artifactId === 'weights');
    assert.deepEqual([...await f.store.readArtifactRange(weight, 0, 1)], [1]);
    assert.deepEqual(f.ranges, [{ artifactId: 'weights', offset: 0, length: 2 }]);
    assert(!f.reads.includes('weights'), 'selective opening must not read the entire shard');
    assert.equal(f.store.getPieceVerification().verifiedBytes, 2);
    f.corrupt();
    await assert.rejects(f.store.readArtifactRange(weight, 2, 1), /integrity/);
  } finally { await f.store.close(); }
}
for (const change of [
  manifest => { manifest.hashAlgorithm = 'sha1'; },
  manifest => { manifest.shards[0].hash = 'malformed'; },
  (_, index) => { index.manifestIdentity = 'sha256:' + '0'.repeat(64); },
  (_, index) => { index.files[0].pieces.pop(); },
  manifest => { manifest.shards[0].size++; },
]) {
  const f = await fixture('blake3', change);
  try { await assert.rejects(f.store.enableResidentPieces(), /hash|Piece|piece/); assert.deepEqual(f.ranges, []); }
  finally { await f.store.close(); }
}
console.log('resident-piece-blake3: SHA256/BLAKE3 selective integrity and five metadata rejections passed');
