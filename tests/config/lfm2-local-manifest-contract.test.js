import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { validateManifestInference } from '../../src/config/schema/manifest.schema.js';

function readJson(path) {
  return JSON.parse(readFileSync(path, 'utf8'));
}

const manifest = readJson('models/local/lfm2-5-1-2b-instruct-q4k-ehf16-af32/manifest.json');
const conversionConfig = readJson('src/config/conversion/lfm2/lfm2.5-1.2b-instruct-q4k-ehf16-af32.json');

assert.doesNotThrow(
  () => validateManifestInference(manifest),
  'lfm2 local manifest must satisfy required inference-field validation'
);

assert.deepEqual(
  manifest.inference?.session?.perLayerInputs,
  conversionConfig.session?.perLayerInputs,
  'lfm2 local manifest must mirror conversion-authored per-layer input policy'
);

for (const kernelKey of ['attn_decode', 'attn_small']) {
  assert.deepEqual(
    manifest.inference?.execution?.kernels?.[kernelKey]?.precision,
    conversionConfig.execution?.kernels?.[kernelKey]?.precision,
    `lfm2 local manifest must mirror ${kernelKey} precision`
  );
}

console.log('lfm2-local-manifest-contract.test: ok');
