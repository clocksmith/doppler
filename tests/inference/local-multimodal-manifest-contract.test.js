// Local multimodal manifests must resolve through the runtime's vision/audio
// config resolvers; text-only manifests must resolve to no vision/audio
// pipeline. Malformed multimodal metadata must fail closed. Local artifacts
// under models/local are gitignored, so absent manifests are skipped.
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';

import {
  resolveAudioConfig,
  resolveVisionConfig,
} from '../../src/inference/pipelines/text/config/normalization.js';

const LOCAL_MANIFESTS = [
  'models/local/qwen-3-5-0-8b-q4k-ehaf16/manifest.json',
  'models/local/qwen-3-5-2b-q4k-ehaf16/manifest.json',
  'models/local/gemma-4-e2b-it-q4k-ehf16-af32/manifest.json',
];

let checked = 0;
let multimodalSample = null;
for (const path of LOCAL_MANIFESTS) {
  if (!existsSync(path)) {
    console.log(`local-multimodal-manifest-contract.test: skipped ${path} (local model missing)`);
    continue;
  }
  checked += 1;
  const manifest = JSON.parse(readFileSync(path, 'utf8'));
  const label = manifest.modelId;
  const rawVision = manifest.config?.vision_config;
  const rawAudio = manifest.config?.audio_config;

  const vision = resolveVisionConfig(manifest.config, manifest);
  if (rawVision == null) {
    assert.equal(vision, null, `${label}: no vision_config must disable the vision pipeline`);
    assert.equal(manifest.image_token_id, undefined, `${label}: image_token_id without vision_config is unusable`);
  } else {
    assert.ok(vision, `${label}: vision_config must resolve`);
    if (vision.minPixels != null || vision.maxPixels != null) {
      assert.ok(vision.minPixels <= vision.maxPixels, `${label}: min_pixels must not exceed max_pixels`);
    }
    if (vision.imageTokenId != null) {
      assert.ok(
        Number.isInteger(vision.imageTokenId) && vision.imageTokenId >= 0 && vision.imageTokenId < manifest.architecture.vocabSize,
        `${label}: image_token_id must be a vocabulary id`
      );
    }
    multimodalSample ??= manifest;
  }

  const audio = resolveAudioConfig(manifest.config, manifest);
  if (rawAudio == null) {
    assert.equal(audio, null, `${label}: no audio_config must disable the audio pipeline`);
  } else {
    assert.equal(audio.audioArchitecture, rawAudio.audio_architecture.trim(), `${label}: audio_config must resolve`);
  }
}

// Malformed metadata fails closed through the same resolvers.
if (multimodalSample) {
  const unsupported = structuredClone(multimodalSample);
  unsupported.config.vision_config.vision_architecture = 'unknown-vision';
  assert.throws(() => resolveVisionConfig(unsupported.config, unsupported), /unsupported vision_config\.vision_architecture/);

  const missingArchitecture = structuredClone(multimodalSample);
  delete missingArchitecture.config.vision_config.vision_architecture;
  assert.throws(() => resolveVisionConfig(missingArchitecture.config, missingArchitecture), /missing vision_config\.vision_architecture/);
}
assert.throws(
  () => resolveAudioConfig({ audio_config: { audio_architecture: 'unknown-audio' } }, { modelId: 'synthetic' }),
  /unsupported audio_config\.audio_architecture/
);

console.log(`local-multimodal-manifest-contract.test: ok (${checked} local manifests)`);
