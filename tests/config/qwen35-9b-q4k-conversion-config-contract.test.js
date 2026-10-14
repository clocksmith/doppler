import assert from 'node:assert/strict';
import fs from 'node:fs';
import { detectSandwichNorm } from '../../src/inference/pipelines/text/layer.js';

function readJson(path) {
  return JSON.parse(fs.readFileSync(path, 'utf8'));
}

const config = readJson('src/config/conversion/qwen3/qwen-3-5-9b-q4k-ehaf16.json');
const f16Config = readJson('src/config/conversion/qwen3/qwen-3-5-9b-f16-af32.json');
const identity = config.manifest?.artifactIdentity;
const f16Identity = f16Config.manifest?.artifactIdentity;

assert.equal(
  identity?.sourceCheckpointId,
  f16Identity?.sourceCheckpointId,
  'Q4K and f16 lanes must convert the same source checkpoint'
);
assert.equal(
  identity?.sourceRevision,
  f16Identity?.sourceRevision,
  'Q4K and f16 lanes must convert the same source revision'
);
assert.deepEqual(
  config.inference?.layerPattern?.layerTypes,
  f16Config.inference?.layerPattern?.layerTypes,
  'Q4K and f16 lanes must share the source checkpoint hybrid schedule'
);

assert.equal(
  detectSandwichNorm(config.inference?.normalization).useSandwichNorm,
  false,
  'Qwen tensor presence must retain the standard residual-then-post-attention-norm route'
);

console.log('qwen35-9b-q4k-conversion-config-contract.test: ok');
