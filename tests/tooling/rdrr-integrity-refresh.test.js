import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import {
  normalizeManifestLoweringEntry,
  refreshManifestIntegrity,
} from '../../src/tooling/rdrr-integrity-refresh.js';
import { runNodeCommand } from '../../src/tooling/node-command-runner.js';

const canonicalManifest = JSON.parse(
  readFileSync(
    new URL('../../models/local/gemma-3-1b-it-q4k-ehf16-af32/manifest.json', import.meta.url),
    'utf8'
  )
);
const tmpDir = await mkdtemp(path.join(os.tmpdir(), 'rdrr-integrity-refresh-'));

try {
  const shard = Buffer.from('0123456789abcdef', 'utf8');
  const manifestPath = path.join(tmpDir, 'manifest.json');
  const loweringEntryPath = path.join(tmpDir, 'fused_gemv.webgpu-generic.json');
  const doeLoweringEntry = {
    backend: 'webgpu-generic',
    compilerVersion: 'doe-tsir-bootstrap-2026-04-24',
    emitterDigest: '3'.repeat(64),
    exactness: {
      algorithmExactInvariants: ['reduction_order', 'accum_dtype'],
      class: 'algorithm_exact',
      toleranceEpsilon: 0,
      toleranceMetric: '',
    },
    frontendVersion: 'frontend-bootstrap-pipeline-v1',
    kernelRef: 'doe.tsir.bootstrap.fused_gemv',
    rejectionReasons: [],
    targetDescriptorCorrectnessHash: '1'.repeat(64),
    tsirRealizationDigest: '2'.repeat(64),
    tsirSemanticDigest: '4'.repeat(64),
  };
  await writeFile(path.join(tmpDir, 'shard_00000.bin'), shard);
  await writeFile(loweringEntryPath, JSON.stringify(doeLoweringEntry), 'utf8');
  await writeFile(
    manifestPath,
    JSON.stringify({
      ...canonicalManifest,
      modelId: 'rdrr-refresh-test',
      shards: [
        { index: 0, filename: 'shard_00000.bin', size: shard.byteLength, hash: 'a'.repeat(64), offset: 0 },
      ],
      totalSize: shard.byteLength,
      tensors: {
        alpha: {
          shard: 0,
          offset: 0,
          size: shard.byteLength,
          shape: [4, 4],
          dtype: 'f16',
          role: 'weight',
        },
      },
      metadata: {
        ...canonicalManifest.metadata,
        source: 'test',
        convertedAt: '2026-04-23T00:00:00.000Z',
      },
      inference: {
        ...canonicalManifest.inference,
        normalization: {
          ...canonicalManifest.inference.normalization,
          postAttentionNorm: false,
          preFeedforwardNorm: false,
          postFeedforwardNorm: false,
        },
      },
    }, null, 2),
    'utf8'
  );

  const dryRun = await refreshManifestIntegrity({
    modelDir: tmpDir,
    dryRun: true,
    blockSize: 8,
    loweringEntryPaths: ['fused_gemv.webgpu-generic.json'],
  });
  assert.equal(dryRun.wrote, false);
  assert.equal(dryRun.manifest.integrityExtensions.blockMerkle.blockSize, 8);
  assert.deepEqual(
    dryRun.manifest.integrityExtensions.lowerings.entries[0],
    normalizeManifestLoweringEntry(doeLoweringEntry)
  );
  assert.equal(
    dryRun.manifest.integrityExtensions.lowerings.entries[0].exactness.class,
    'algorithm_exact'
  );

  const written = await refreshManifestIntegrity({
    modelDir: tmpDir,
    blockSize: 8,
    loweringEntries: [doeLoweringEntry],
  });
  assert.equal(written.wrote, true);
  assert.equal(written.manifest.metadata.integrityRefresh.blockSize, 8);
  const persisted = JSON.parse(await readFile(manifestPath, 'utf8'));
  assert.equal(persisted.integrityExtensions.blockMerkle.blockSize, 8);
  assert.equal(
    persisted.integrityExtensions.lowerings.entries[0].targetDescriptorCorrectnessHash,
    '1'.repeat(64)
  );
  assert.ok(persisted.metadata.integrityRefresh.at);

  const commandResult = await runNodeCommand({
    command: 'refresh-integrity',
    modelDir: tmpDir,
    blockSize: 8,
    dryRun: true,
  });
  assert.equal(commandResult.ok, true);
  assert.equal(commandResult.surface, 'node');
  assert.equal(commandResult.request.command, 'refresh-integrity');
  assert.equal(commandResult.result.wrote, false);
  assert.equal(commandResult.result.integrityExtensions.blockMerkle.blockSize, 8);
} finally {
  await rm(tmpDir, { recursive: true, force: true });
}

console.log('rdrr-integrity-refresh.test: ok');
