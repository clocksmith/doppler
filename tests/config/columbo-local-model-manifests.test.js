import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { validateManifestInference } from '../../src/config/schema/manifest.schema.js';

function readJson(relativePath) {
  return JSON.parse(readFileSync(path.join(process.cwd(), relativePath), 'utf8'));
}

const manifestPaths = [
  'models/local/google-embeddinggemma-300m-q4k-ehf16-af32/manifest.json',
  'models/local/qwen-3-5-0-8b-q4k-ehaf16/manifest.json',
  'models/local/qwen-3-5-2b-q4k-ehaf16/manifest.json',
];

for (const manifestPath of manifestPaths) {
  const manifest = readJson(manifestPath);
  assert.doesNotThrow(
    () => validateManifestInference(manifest),
    `${manifest.modelId} must satisfy required inference-field validation`
  );
  if (manifest.modelId === 'google-embeddinggemma-300m-q4k-ehf16-af32') {
    assert.ok(
      manifest.inference?.session?.perLayerInputs,
      'google-embeddinggemma-300m-q4k-ehf16-af32 must stamp inference.session.perLayerInputs for Columbo retrieval'
    );
  }
}

console.log('columbo-local-model-manifests.test: ok');
