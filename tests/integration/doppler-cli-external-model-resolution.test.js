import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const {
  resolveBrowserModelUrl,
  resolveNodeModelUrl,
} = await import('../../src/cli/doppler-cli.js');

const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-cli-external-'));

try {
  const rdrrRoot = path.join(tempRoot, 'rdrr');
  const modelId = 'translategemma-4b-it-q4k-ehf16-af32';
  const staticRoot = path.join(tempRoot, 'app');
  await fs.mkdir(staticRoot, { recursive: true });

  const localModelId = 'qwen-3-5-0-8b-q4k-ehaf16';
  const localModelDir = path.join(staticRoot, 'models', 'local', localModelId);
  await fs.mkdir(localModelDir, { recursive: true });
  await fs.writeFile(path.join(localModelDir, 'manifest.json'), '{}', 'utf8');
  await fs.writeFile(path.join(localModelDir, 'shard_0.bin'), '');

  const localResolved = await resolveNodeModelUrl(
    { modelId: localModelId },
    { staticRootDir: staticRoot, rdrrRoot }
  );
  assert.equal(localResolved.modelId, localModelId);
  assert.equal(localResolved.modelUrl, pathToFileURL(localModelDir).href.replace(/\/$/, ''));

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

  const incompleteLocalModelId = 'translategemma-4b-1b-enes-q4k-ehf16-af32';
  const incompleteLocalModelDir = path.join(staticRoot, 'models', 'local', incompleteLocalModelId);
  await fs.mkdir(incompleteLocalModelDir, { recursive: true });
  await fs.writeFile(path.join(incompleteLocalModelDir, 'manifest.json'), '{}', 'utf8');

  const externalFallbackDir = path.join(rdrrRoot, incompleteLocalModelId);
  await fs.mkdir(externalFallbackDir, { recursive: true });
  await fs.writeFile(path.join(externalFallbackDir, 'manifest.json'), '{}', 'utf8');
  await fs.writeFile(path.join(externalFallbackDir, 'shard_0.bin'), '');

  const externalFallbackResolved = await resolveNodeModelUrl(
    { modelId: incompleteLocalModelId },
    { staticRootDir: staticRoot, rdrrRoot }
  );
  assert.equal(externalFallbackResolved.modelId, incompleteLocalModelId);
  assert.equal(
    externalFallbackResolved.modelUrl,
    pathToFileURL(externalFallbackDir).href.replace(/\/$/, '')
  );

  await assert.rejects(
    () => resolveNodeModelUrl({ modelId: 'missing-model' }, { rdrrRoot }),
    (err) => {
      assert.match(err.message, /Model "missing-model" not found/u);
      assert.match(err.message, /Not in catalog/u);
      assert.match(err.message, /request\.modelUrl/u);
      return true;
    }
  );

  // Catalog fallback: known alias resolves to HF URL when not found locally
  const catalogResolved = await resolveNodeModelUrl(
    { modelId: 'gemma3-270m' },
    { rdrrRoot }
  );
  assert.equal(catalogResolved.modelId, 'gemma3-270m');
  assert.ok(
    catalogResolved.modelUrl.startsWith('https://huggingface.co/Clocksmith/rdrr/resolve/'),
    `Expected HF URL, got: ${catalogResolved.modelUrl}`
  );
  assert.ok(
    catalogResolved.modelUrl.includes('gemma-3-270m-it-q4k-ehf16-af32'),
    `Expected model path in URL, got: ${catalogResolved.modelUrl}`
  );

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

  // Browser catalog fallback: known alias resolves to HF URL when not found locally
  const browserCatalogResolved = await resolveBrowserModelUrl(
    { modelId: 'gemma3-270m' },
    { staticRootDir: staticRoot, rdrrRoot }
  );
  assert.ok(
    browserCatalogResolved.modelUrl.startsWith('https://huggingface.co/Clocksmith/rdrr/resolve/'),
    `Expected HF URL for browser catalog fallback, got: ${browserCatalogResolved.modelUrl}`
  );

  const normalizedModelId = 'sana-sprint-0-6b-f16';
  const aliasedDir = path.join(rdrrRoot, 'sana-sprint-0.6b-f16');
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
