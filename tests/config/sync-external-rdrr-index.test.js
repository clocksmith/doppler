import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { buildIndex } from '../../tools/sync-external-rdrr-index.js';

const root = mkdtempSync(path.join(tmpdir(), 'doppler-external-rdrr-index-'));

try {
  const rdrrRoot = path.join(root, 'rdrr');
  const modelDir = path.join(rdrrRoot, 'demo-model');
  mkdirSync(modelDir, { recursive: true });

  writeFileSync(path.join(modelDir, 'manifest.json'), JSON.stringify({
    modelId: 'demo-model-wq4k-ef16-hf16',
    quantization: 'Q4_K',
    totalSize: 1024,
    shards: [{ filename: 'shard_00000.bin' }],
    metadata: {
      sourceFormat: 'safetensors',
    },
  }, null, 2), 'utf8');

  await assert.rejects(
    () => buildIndex({
      volumeRoot: root,
      rdrrRoot,
      jsonOutput: path.join(root, 'RDRR_INDEX.json'),
      mdOutput: path.join(root, 'RDRR_INDEX.md'),
    }, '2026-03-08T00:00:00.000Z'),
    /Missing explicit sourceModel\/sourceRepo metadata/
  );

  writeFileSync(path.join(modelDir, 'origin.json'), JSON.stringify({
    sourceRepo: 'google/demo-model',
    sourceFormat: 'safetensors',
    variant: 'wq4k-ef16-hf16',
  }, null, 2), 'utf8');

  const outputs = await buildIndex({
    volumeRoot: root,
    rdrrRoot,
    jsonOutput: path.join(root, 'RDRR_INDEX.json'),
    mdOutput: path.join(root, 'RDRR_INDEX.md'),
  }, '2026-03-08T00:00:00.000Z');
  const payload = JSON.parse(outputs.json);
  assert.equal(payload.summary.sourceModelCount, 1);
  assert.equal(payload.summary.variantCount, 1);
  assert.equal(payload.sourceModels[0].sourceModel, 'google/demo-model');
  assert.equal(payload.sourceModels[0].variants[0].variant, 'wq4k-ef16-hf16');
}
finally {
  rmSync(root, { recursive: true, force: true });
}

console.log('sync-external-rdrr-index.test: ok');
