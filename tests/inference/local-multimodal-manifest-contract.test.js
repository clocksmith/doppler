import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

function readManifest(path) {
  return JSON.parse(readFileSync(path, 'utf8'));
}

const qwen08 = readManifest('models/local/qwen-3-5-0-8b-q4k-ehaf16/manifest.json');
assert.equal(qwen08.image_token_id, 248056);
assert.equal(qwen08.config?.vision_config?.vision_architecture, 'qwen3vl');
assert.deepEqual(qwen08.config?.vision_config?.normalization, {
  mean: [0.5, 0.5, 0.5],
  std: [0.5, 0.5, 0.5],
});
assert.equal(qwen08.config?.vision_config?.min_pixels, 65536);
assert.equal(qwen08.config?.vision_config?.max_pixels, 16777216);

const qwen2 = readManifest('models/local/qwen-3-5-2b-q4k-ehaf16/manifest.json');
assert.equal(qwen2.image_token_id, undefined);
assert.equal(qwen2.config?.vision_config, undefined);

const gemma = readManifest('models/local/gemma-4-e2b-it-q4k-ehf16-af32/manifest.json');
assert.equal(gemma.config?.vision_config?.vision_architecture, 'gemma4');
assert.equal(gemma.config?.audio_config?.audio_architecture, 'gemma4');

console.log('local-multimodal-manifest-contract.test: ok');
