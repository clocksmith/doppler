import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  assertLocalModelArtifactsReadable,
  normalizeModelUrl,
  parseArgs,
} from '../../tools/run-program-bundle-reference.js';
import {
  resolveProgramBundleStorageArtifact,
} from '../../src/tooling/program-bundle.js';

const repoRoot = process.cwd();
const tmpRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-program-reference-'));
const variantDir = path.join(tmpRoot, 'models', 'variant');
const weightsDir = path.join(tmpRoot, 'models', 'weights');

async function writeJson(filePath, payload) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
}

try {
  const storageManifest = {
    version: 1,
    modelId: 'weights',
    tokenizer: {
      type: 'bundled',
      file: 'tokenizer.json',
    },
    shards: [
      {
        index: 0,
        filename: 'shard_00000.bin',
        size: 4,
        hash: '1'.repeat(64),
        offset: 0,
      },
    ],
    artifactIdentity: {
      weightPackId: 'toy-wp',
      shardSetHash: `sha256:${'a'.repeat(64)}`,
    },
  };
  const variantManifest = {
    version: 1,
    modelId: 'variant',
    weightsRef: {
      weightPackId: 'toy-wp',
      artifactRoot: '../weights',
      shardSetHash: `sha256:${'a'.repeat(64)}`,
    },
    tokenizer: {
      type: 'bundled',
      file: 'tokenizer.json',
    },
    shards: [
      {
        index: 0,
        filename: 'missing-from-variant.bin',
        size: 4,
        hash: '2'.repeat(64),
        offset: 0,
      },
    ],
  };

  await writeJson(path.join(weightsDir, 'manifest.json'), storageManifest);
  await fs.writeFile(path.join(weightsDir, 'tokenizer.json'), '{}\n', 'utf8');
  await fs.writeFile(path.join(weightsDir, 'shard_00000.bin'), Buffer.alloc(4));
  await writeJson(path.join(variantDir, 'manifest.json'), variantManifest);

  await assertLocalModelArtifactsReadable({
    modelUrl: pathToFileURL(variantDir).href,
    manifest: variantManifest,
  });

  await assertLocalModelArtifactsReadable({
    modelUrl: '/models/variant',
    localArtifactModelDir: variantDir,
    manifest: variantManifest,
  });

  const storageArtifact = await resolveProgramBundleStorageArtifact(variantManifest, variantDir);
  assert.equal(storageArtifact.modelDir, weightsDir);
  assert.equal(storageArtifact.manifest.modelId, 'weights');
  assert.equal(storageArtifact.manifestPath, path.join(weightsDir, 'manifest.json'));

  await fs.rm(path.join(weightsDir, 'shard_00000.bin'));
  await assert.rejects(
    () => assertLocalModelArtifactsReadable({
      modelUrl: pathToFileURL(variantDir).href,
      manifest: variantManifest,
    }),
    /models\/weights\/shard_00000\.bin/
  );

  const repoLocalDir = path.join(repoRoot, 'models', 'local', 'qwen-3-6-27b-q4k-eaf16');
  assert.equal(
    normalizeModelUrl('', repoLocalDir, { repoRoot, surface: 'browser' }),
    '/models/local/qwen-3-6-27b-q4k-eaf16'
  );
  assert.match(
    normalizeModelUrl('', repoLocalDir, { repoRoot, surface: 'node' }),
    /^file:\/\//
  );

  assert.throws(
    () => parseArgs([
      '--surface',
      'browser',
      '--manifest',
      path.join(variantDir, 'manifest.json'),
      '--out',
      path.join(tmpRoot, 'bundle.json'),
      '--tsir-fixture-dir',
      path.join(tmpRoot, 'fixture'),
    ]),
    /--tsir-fixture-dir requires --surface node/
  );
} finally {
  await fs.rm(tmpRoot, { recursive: true, force: true });
}

console.log('program-bundle-reference-preflight.test: ok');
