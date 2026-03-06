import assert from 'node:assert/strict';
import fs from 'node:fs';

import {
  inferConversionConfigModelId,
  resolveMaterializedManifestFromConversionConfig,
} from '../../src/tooling/conversion-config-materializer.js';

const conversionConfig = JSON.parse(
  fs.readFileSync('tools/configs/conversion/gemma3/gemma-3-270m-it-wq4k-ef16-hf16.json', 'utf8')
);
const manifest = JSON.parse(
  fs.readFileSync('models/local/gemma-3-270m-it-wq4k-ef16-hf16/manifest.json', 'utf8')
);

assert.equal(
  inferConversionConfigModelId(
    'tools/configs/conversion/gemma3/gemma-3-270m-it-wq4k-ef16-hf16.json',
    conversionConfig
  ),
  'gemma-3-270m-it-wq4k-ef16-hf16'
);

const materialized = resolveMaterializedManifestFromConversionConfig(conversionConfig, manifest);
assert.equal(materialized.modelId, manifest.modelId);
assert.equal(materialized.modelType, 'transformer');
assert.equal(materialized.inference?.schema, 'doppler.execution/v0');
assert.equal(materialized.inference?.defaultKernelPath, 'gemma3-q4k-dequant-f32a-online');
assert.ok(Array.isArray(materialized.inference?.execution?.steps));
assert.ok(materialized.inference.execution.steps.length > 0);

console.log('conversion-config-materializer.test: ok');
