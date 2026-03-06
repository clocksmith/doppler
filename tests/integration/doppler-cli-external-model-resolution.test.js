import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const {
  resolveBrowserModelUrl,
  resolveNodeModelUrl,
} = await import('../../tools/doppler-cli.js');

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

  const staticRoot = path.join(tempRoot, 'app');
  await fs.mkdir(staticRoot, { recursive: true });

  const browserResolved = await resolveBrowserModelUrl(
    { modelId },
    { staticRootDir: staticRoot, rdrrRoot }
  );
  assert.equal(browserResolved.modelId, modelId);
  assert.equal(browserResolved.modelUrl, `/models/external/${encodeURIComponent(modelId)}`);

  const browserUnchanged = await resolveBrowserModelUrl(
    { modelId: 'missing-model' },
    { staticRootDir: staticRoot, rdrrRoot }
  );
  assert.equal(browserUnchanged.modelUrl, '/models/missing-model');

  const normalizedModelId = 'sana-sprint-0-6b-wf16-ef16-hf16-f16';
  const aliasedDir = path.join(rdrrRoot, 'sana-sprint-0.6b-wf16-ef16-hf16-f16');
  await fs.mkdir(aliasedDir, { recursive: true });
  await fs.writeFile(path.join(aliasedDir, 'manifest.json'), JSON.stringify({ modelId: normalizedModelId }), 'utf8');
  await fs.writeFile(path.join(aliasedDir, 'shard_0.bin'), '');

  const nodeResolvedByManifestId = await resolveNodeModelUrl(
    { modelId: normalizedModelId },
    { rdrrRoot }
  );
  assert.equal(nodeResolvedByManifestId.modelId, normalizedModelId);
  assert.equal(nodeResolvedByManifestId.modelUrl, pathToFileURL(aliasedDir).href.replace(/\/$/, ''));

  const browserResolvedByManifestId = await resolveBrowserModelUrl(
    { modelId: normalizedModelId },
    { staticRootDir: staticRoot, rdrrRoot }
  );
  assert.equal(
    browserResolvedByManifestId.modelUrl,
    `/models/external/${encodeURIComponent(path.basename(aliasedDir))}`
  );
} finally {
  await fs.rm(tempRoot, { recursive: true, force: true });
}

console.log('doppler-cli-external-model-resolution.test: ok');
