import assert from 'node:assert/strict';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { computeHash } from '../../src/storage/shard-manager.js';
import { loadCapsulePieceIndex } from '../../src/tooling/model-capsule-pieces.js';

const directory = await mkdtemp(join(tmpdir(), 'capsule-pieces-'));
const bytes = Uint8Array.of(97, 98, 99, 100, 101);
const sha = value => 'sha256:' + createHash('sha256').update(value).digest('hex');
async function fixture(algorithm, declaredHash = null) {
  const manifest = { hashAlgorithm: algorithm, shards: [{ filename: 'weights.bin', size: bytes.length,
    hash: declaredHash ?? await computeHash(bytes, algorithm) }] };
  const raw = JSON.stringify(manifest);
  const index = { schema: 'doppler.verified-pieces/v1', manifestIdentity: sha(raw), maxPieceBytes: 3,
    files: [{ path: 'weights.bin', size: bytes.length, pieces: [
      { offset: 0, size: 3, identity: sha(bytes.subarray(0, 3)) },
      { offset: 3, size: 2, identity: sha(bytes.subarray(3)) },
    ] }] };
  await writeFile(join(directory, 'weights.bin'), bytes);
  await writeFile(join(directory, 'pieces.json'), JSON.stringify(index));
  return raw;
}
try {
  for (const algorithm of ['sha256', 'blake3']) {
    const raw = await fixture(algorithm);
    const result = await loadCapsulePieceIndex(join(directory, 'pieces.json'), raw, directory);
    assert.equal(result.path, 'artifacts/model/pieces.json');
    const wrongHash = await fixture(algorithm, '0'.repeat(64));
    await assert.rejects(loadCapsulePieceIndex(join(directory, 'pieces.json'), wrongHash, directory),
      /shard.*hash mismatch/i, 'Individually valid pieces must also match their declared whole shard');
    await fixture(algorithm);
    await writeFile(join(directory, 'weights.bin'), Uint8Array.of(97, 98, 99, 100, 102));
    await assert.rejects(loadCapsulePieceIndex(join(directory, 'pieces.json'), raw, directory), /integrity mismatch/);
  }
  const raw = await fixture('sha1', '0'.repeat(40));
  await assert.rejects(loadCapsulePieceIndex(join(directory, 'pieces.json'), raw, directory), /unsupported.*hash/i);
  console.log('model-capsule-pieces: 7 source integrity controls passed');
} finally {
  await rm(directory, { recursive: true, force: true });
}
