import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

// Applications that imported these modules by internal path can switch to the
// package entry points; each entry must expose the same implementation object.
const { exports: packageExports } = JSON.parse(readFileSync('package.json', 'utf8'));

async function importEntry(subpath) {
  const target = packageExports[subpath];
  assert.ok(target, `package.json must export ${subpath}`);
  return import(pathToFileURL(target.import).href);
}

const migrations = [
  ['./tooling/storage', 'src/storage/shard-manager.js', ['openModelStore', 'modelExists', 'loadManifestFromStore', 'loadTokenizerFromStore', 'verifyIntegrity', 'deleteModel']],
  ['./tooling/storage', 'src/storage/model-cache.js', ['loadPersistentModelSource']],
  ['./tooling/storage', 'src/storage/source-artifact-store.js', ['verifyStoredSourceArtifact']],
  ['./tooling/storage', 'src/tooling/opfs-cache.js', ['ensureModelCachedSource']],
  ['./tooling/storage', 'src/storage/download/resume-inspection.js', ['buildManifestVersionSet', 'inspectModelDownloadResume']],
  ['./tooling/device', 'src/gpu/device.js', ['initDevice', 'getKernelCapabilities', 'destroyDevice']],
  ['./tooling/manifest', 'src/formats/rdrr/index.js', ['parseManifest']],
  ['./tooling/manifest', 'src/config/schema/index.js', ['DEFAULT_MANIFEST_INFERENCE', 'DEFAULT_KVCACHE_CONFIG']],
  ['./tooling/runtime', 'src/tooling/test-harness.js', ['initializeInference']],
  ['./tooling', 'src/tooling/browser-harness-text-helpers.js', ['runRerank']],
  ['./generation', 'src/inference/pipelines/text.js', ['createPipeline', 'AbortError', 'isAbortError']],
  ['./structured', 'src/inference/pipelines/structured/json-grammar-mask.js', ['createJsonGrammarMask']],
  ['./training', 'src/experimental/adapters/lora-loader.js', ['loadLoRAFromManifest']],
];

for (const [subpath, implementationPath, names] of migrations) {
  const entry = await importEntry(subpath);
  const implementation = await import(pathToFileURL(implementationPath).href);
  for (const name of names) {
    assert.ok(name in implementation, `${implementationPath} must export ${name}`);
    assert.equal(entry[name], implementation[name], `${subpath} ${name} must be the ${implementationPath} implementation`);
  }
}

// BLAKE3 is exposed under an unambiguous storage name.
{
  const storage = await importEntry('./tooling/storage');
  const blake3 = await import(pathToFileURL('src/storage/blake3.js').href);
  assert.equal(storage.createBlake3Hasher, blake3.createHasher);
  const bytes = new TextEncoder().encode('doppler');
  const hasher = storage.createBlake3Hasher();
  hasher.update(bytes);
  assert.deepEqual(hasher.finalize(), await blake3.hash(bytes));
}

// The browser tooling surface carries the same shared additions.
{
  const browserTooling = await import(pathToFileURL('src/tooling-exports.browser.js').href);
  const nodeTooling = await importEntry('./tooling');
  for (const name of ['runRerank', 'modelExists', 'verifyIntegrity', 'destroyDevice', 'DEFAULT_KVCACHE_CONFIG']) {
    assert.equal(browserTooling[name], nodeTooling[name], `${name} must match across tooling surfaces`);
  }
}

console.log('consumer-entry-points.test: ok');
