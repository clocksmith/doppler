import assert from 'node:assert/strict';
import fs from 'node:fs';

const MODEL_ID = 'gemma-4-12b-it-text-q4k-ehf16-af32';
const CONFIG_PATH = `src/config/conversion/gemma4/${MODEL_ID}.json`;

const conversionConfig = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8'));
const layerTypes = conversionConfig.inference?.layerPattern?.layerTypes ?? [];
const fullAttentionLayers = layerTypes
  .map((kind, index) => (kind === 'full_attention' ? index : null))
  .filter((index) => index != null);

assert.ok(fullAttentionLayers.length > 0, 'layerPattern must declare full-attention layers');
assert.deepEqual(
  conversionConfig.execution?.prefill?.[1]?.layers,
  fullAttentionLayers,
  'the full-attention prefill group must cover exactly the layerPattern full-attention layers'
);

console.log('gemma4-12b-conversion-config-contract.test: ok');
