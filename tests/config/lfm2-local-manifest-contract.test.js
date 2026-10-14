import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { validateManifestInference } from '../../src/config/schema/manifest.schema.js';

function readJson(path) {
  return JSON.parse(readFileSync(path, 'utf8'));
}

const manifest = readJson('models/local/lfm2-5-1-2b-instruct-q4k-ehf16-af32/manifest.json');

assert.doesNotThrow(
  () => validateManifestInference(manifest),
  'lfm2 local manifest must satisfy required inference-field validation'
);

console.log('lfm2-local-manifest-contract.test: ok');
