import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { createDopplerConfig } from '../../src/config/schema/doppler.schema.js';
import { requireDeclaredKVMaxSeqLen } from '../../src/config/schema/kvcache.schema.js';
import {
  applyModelSessionDeclaration,
  listModelSessionDeclarations,
} from '../../src/config/model-session-declarations.js';
import { applyModelBatchingRuntimeDefaults } from '../../src/inference/pipelines/text/model-load.js';

// Every tracked manifest, catalog entry and quickstart model must resolve a KV
// length without a runtime override, so a missing declaration fails here rather
// than at a user's load.
const readJson = (file) => JSON.parse(readFileSync(file, 'utf8'));
const runtimeConfig = createDopplerConfig().runtime;

// Catalog entries whose manifests are published but not tracked here. Values
// were read from the hosted manifests at the catalog revision; adding another
// remote-only entry must extend this list after checking its manifest.
const REMOTE_ONLY_DECLARED = new Map([
  ['amplify-120m-f16-af32', 2048],
  ['esm2-t12-35m-ur50d-f32-af32', 1026],
  ['esmc-300m-f32-af32', 2048],
  ['nucleotide-transformer-v2-50m-f32-af32', 2050],
]);

const manifestFiles = execFileSync('git', ['ls-files', 'models/local/*/manifest.json'], { encoding: 'utf8' })
  .split('\n').filter(Boolean);
assert.ok(manifestFiles.length > 0, 'expected tracked manifests under models/local');

const resolvedByModelId = new Map();
const usedDeclarations = new Set();
for (const file of manifestFiles) {
  const manifest = readJson(file);
  const manifestDeclared = manifest.inference?.session?.kvcache?.maxSeqLen;
  if (Number.isSafeInteger(manifestDeclared) && manifestDeclared > 0) {
    assert.ok(
      !listModelSessionDeclarations().includes(manifest.modelId),
      `${file}: manifest declares maxSeqLen ${manifestDeclared}; remove "${manifest.modelId}" from model-sessions.json`
    );
  } else {
    usedDeclarations.add(manifest.modelId);
  }
  const contextLength = manifest.architecture?.maxSeqLen;
  const sessionKV = applyModelSessionDeclaration(manifest).kvcache;
  const declared = requireDeclaredKVMaxSeqLen(sessionKV?.maxSeqLen);
  assert.ok(declared <= contextLength, `${file}: KV length ${declared} exceeds context ${contextLength}`);
  if (!(Number.isSafeInteger(manifestDeclared) && manifestDeclared > 0)) {
    assert.equal(declared, Math.min(4096, contextLength), `${file}: declaration must keep the previous min(4096, context)`);
  }
  // The full session merge (no runtime override) carries the same value.
  if (manifest.inference?.session?.decodeLoop) {
    const merged = applyModelBatchingRuntimeDefaults(runtimeConfig, manifest, null);
    assert.equal(merged.inference.session.kvcache.maxSeqLen, declared, `${file}: merged session KV length`);
  }
  resolvedByModelId.set(manifest.modelId, declared);
}

const catalogIds = readJson('models/catalog.json').models.map((entry) => entry.modelId);
const quickstartIds = readJson('src/config/quickstart-registry.json').models.map((entry) => entry.modelId);
for (const modelId of new Set([...catalogIds, ...quickstartIds])) {
  if (resolvedByModelId.has(modelId)) continue;
  assert.ok(
    REMOTE_ONLY_DECLARED.has(modelId),
    `${modelId} has no tracked manifest and no recorded manifest KV declaration; check its published manifest`
  );
}
for (const modelId of REMOTE_ONLY_DECLARED.keys()) {
  assert.ok(catalogIds.includes(modelId), `REMOTE_ONLY_DECLARED lists ${modelId}, which is not in the catalog`);
  assert.ok(!listModelSessionDeclarations().includes(modelId), `${modelId} declares in its manifest`);
}

// Declarations for untracked published artifacts must still be bound to a context.
for (const modelId of listModelSessionDeclarations()) {
  if (usedDeclarations.has(modelId)) continue;
  assert.equal(modelId, 'gemma-3-1b-it-f16-af32', `unexpected declaration without a tracked manifest: ${modelId}`);
}

// A runtime override still wins over the declaration.
const quickstartManifest = readJson('models/local/gemma-3-270m-it-q4k-ehf16-af32/manifest.json');
const overridden = applyModelBatchingRuntimeDefaults(
  { ...runtimeConfig, inference: { ...runtimeConfig.inference, session: {
    ...runtimeConfig.inference.session, kvcache: { ...runtimeConfig.inference.session.kvcache, maxSeqLen: 1024 },
  } } },
  quickstartManifest,
  null
);
assert.equal(overridden.inference.session.kvcache.maxSeqLen, 1024);

// A declaration does not apply to an artifact with a different context.
assert.throws(
  () => applyModelSessionDeclaration({
    ...quickstartManifest,
    architecture: { ...quickstartManifest.architecture, maxSeqLen: 8192 },
  }),
  /declares "gemma-3-270m-it-q4k-ehf16-af32" for contextLength 32768.*8192/
);

// Unlisted models still fail fast.
assert.throws(
  () => requireDeclaredKVMaxSeqLen(applyModelSessionDeclaration({ ...quickstartManifest, modelId: 'unlisted' }).kvcache?.maxSeqLen),
  /not declared/
);

console.log('model-kv-declaration-contract.test: ok');
