import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createDopplerRuntimeService } from '../../src/client/model-host/index.js';
import { resolveModelSource } from '../../src/client/runtime/model-source.js';
import { resolveNodeModelUrl } from '../../src/cli/cli-model-resolution.js';
import { buildQuickstartRegistryPayload } from '../../tools/sync-quickstart-registry.js';

const catalog = JSON.parse(await readFile(new URL('../../models/catalog.json', import.meta.url)));
const revocations = JSON.parse(await readFile(new URL('../../src/config/revocation-registry.json', import.meta.url)));
const entry = catalog.models.find((model) => model.quickstart && model.hostedSource);
assert.ok(entry);
const oldId = entry.hostedSource.weights.path.split('/').at(-1);
const oldText = await readFile(new URL(`../../models/local/${oldId}/manifest.json`, import.meta.url), 'utf8');
const oldManifest = JSON.parse(oldText);
assert.equal(oldManifest.modelId, oldId);
const wrongText = JSON.stringify({ ...oldManifest, modelId: entry.modelId });
assert.notEqual(createHash('sha256').update(wrongText).digest('hex'), entry.hostedSource.expectedManifestHash);

const runtime = createDopplerRuntimeService({ ensureWebGPUAvailable: async () => {} });
const calls = [];
const savedFetch = globalThis.fetch;
const tmp = await mkdtemp(path.join(os.tmpdir(), 'doppler-named-source-'));
try {
  const source = await resolveModelSource(entry.modelId);
  assert.equal(source.modelId, entry.modelId);
  assert.equal(source.manifestUrl, entry.hostedSource.manifestUrl);
  assert.equal(source.expectedManifestHash, entry.hostedSource.expectedManifestHash);
  assert.ok(source.baseUrl.endsWith(entry.hostedSource.weights.path));
  const legacy = await resolveModelSource(oldId);
  assert.equal(legacy.modelId, oldId);
  assert.equal(legacy.manifestUrl, undefined);

  for (const [text, status] of [[oldText, 200], [wrongText, 200], ['', 404]]) {
    calls.length = 0;
    globalThis.fetch = async (url) => {
      calls.push(String(url));
      return new Response(text, { status });
    };
    await assert.rejects(runtime.load(entry.modelId), status === 404 ? /404/ : /Manifest identity mismatch/);
    assert.deepEqual(calls, [entry.hostedSource.manifestUrl]);
  }

  const directory = path.join(tmp, entry.modelId);
  await mkdir(directory);
  await writeFile(path.join(directory, 'manifest.json'), wrongText);
  const localUrl = pathToFileURL(directory).href;
  calls.length = 0;
  await assert.rejects(runtime.load({ registryId: entry.modelId, url: localUrl }), /Manifest identity mismatch/);
  await assert.rejects(resolveNodeModelUrl({ modelId: entry.modelId, modelUrl: localUrl }), /Manifest identity mismatch/);
  assert.deepEqual(calls, []);

  // Concurrent convenience calls to distinct overrides each inspect their own
  // manifest even though they share a catalog ID and model weights.
  const missingUrl = pathToFileURL(path.join(tmp, 'absent-directory')).href;
  const attempts = await Promise.allSettled([
    runtime.getCachedModel({ registryId: entry.modelId, url: localUrl }),
    runtime.getCachedModel({ registryId: entry.modelId, url: missingUrl }),
  ]);
  assert(attempts.every(result => result.status === 'rejected'));
  assert.match(attempts[0].reason.message, /Manifest identity mismatch/);
  assert.match(String(attempts[1].reason), /ENOENT|not found|absent-directory/i);

  for (const invalid of [
    { ...entry.hostedSource, expectedManifestHash: '' },
    { ...entry.hostedSource, manifestUrl: 'http://example.org/manifest.json' },
    { ...entry.hostedSource, weights: { ...entry.hostedSource.weights, revision: 'main' } },
  ]) {
    assert.throws(() => buildQuickstartRegistryPayload({ models: [{ ...entry, hostedSource: invalid }] }, revocations),
      /separate manifest and weight sources/);
  }
} finally {
  globalThis.fetch = savedFetch;
  await rm(tmp, { recursive: true, force: true });
}

console.log('named-quickstart-source.test: ok');
