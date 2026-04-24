import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { buildMerkleTree } from '../../src/formats/rdrr/merkle.js';
import { buildManifestIntegrityFromModelDir } from '../../src/tooling/rdrr-integrity-refresh.js';

const canonicalManifest = JSON.parse(
  readFileSync(
    new URL('../../models/local/gemma-3-1b-it-q4k-ehf16-af32/manifest.json', import.meta.url),
    'utf8'
  )
);
const tmpDir = await mkdtemp(path.join(os.tmpdir(), 'rdrr-integrity-'));

try {
  const shard0 = Buffer.from('abcdefghijklmnop', 'utf8');
  const shard1 = Buffer.from('qrstuvwxyz', 'utf8');
  await writeFile(path.join(tmpDir, 'shard_00000.bin'), shard0);
  await writeFile(path.join(tmpDir, 'shard_00001.bin'), shard1);

  const manifest = {
    ...canonicalManifest,
    modelId: 'rdrr-integrity-test',
    shards: [
      { index: 0, filename: 'shard_00000.bin', size: shard0.byteLength, hash: 'sha256:a', offset: 0 },
      { index: 1, filename: 'shard_00001.bin', size: shard1.byteLength, hash: 'sha256:b', offset: shard0.byteLength },
    ],
    totalSize: shard0.byteLength + shard1.byteLength,
    tensors: {
      alpha: {
        shard: 0,
        offset: 2,
        size: 8,
        shape: [2, 4],
        dtype: 'f16',
        role: 'weight',
      },
      beta: {
        spans: [
          { shard: 0, offset: 12, size: 4 },
          { shard: 1, offset: 0, size: 6 },
        ],
        size: 10,
        shape: [2, 5],
        dtype: 'f16',
        role: 'weight',
      },
    },
  };

  const built = await buildManifestIntegrityFromModelDir(manifest, {
    modelDir: tmpDir,
    blockSize: 4,
  });

  assert.equal(built.integrityExtensions.blockMerkle.blockSize, 4);
  assert.equal(built.integrityExtensionsHash.startsWith('integrity:sha256:'), true);
  assert.deepEqual(
    built.integrityExtensions.blockMerkle.roots,
    {
      alpha: buildMerkleTree(shard0.subarray(2, 10), { blockSize: 4 }).root,
      beta: buildMerkleTree(Buffer.concat([shard0.subarray(12, 16), shard1.subarray(0, 6)]), { blockSize: 4 }).root,
    }
  );
} finally {
  await rm(tmpDir, { recursive: true, force: true });
}

console.log('rdrr-integrity.test: ok');
