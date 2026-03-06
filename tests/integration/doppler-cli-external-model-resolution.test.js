import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const { resolveNodeModelUrl } = await import('../../tools/doppler-cli.js');

const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-cli-external-'));

try {
  const rdrrRoot = path.join(tempRoot, 'rdrr');
  const modelId = 'translategemma-4b-it-wq4k-ef16-hf16';
  const modelDir = path.join(rdrrRoot, modelId);
  await fs.mkdir(modelDir, { recursive: true });
  await fs.writeFile(path.join(modelDir, 'manifest.json'), '{}', 'utf8');
  await fs.writeFile(path.join(modelDir, 'shard_0.bin'), '');

  const resolved = await resolveNodeModelUrl(
    { modelId },
    { rdrrRoot }
  );

  assert.equal(resolved.modelId, modelId);
  assert.equal(resolved.modelUrl, pathToFileURL(modelDir).href.replace(/\/$/, ''));

  const unchanged = await resolveNodeModelUrl(
    { modelId: 'missing-model' },
    { rdrrRoot }
  );
  assert.equal(unchanged.modelUrl, undefined);
} finally {
  await fs.rm(tempRoot, { recursive: true, force: true });
}

console.log('doppler-cli-external-model-resolution.test: ok');
