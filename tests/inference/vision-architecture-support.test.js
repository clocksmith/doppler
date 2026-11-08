import assert from 'node:assert/strict';
import { resolveVisionArchitectureSupport, executableVisionArchitectures, describeVisionUnavailable } from '../../src/inference/pipelines/vision/support.js';
import { encodeImage } from '../../src/inference/pipelines/vision/index.js';
import { _ensureVisionWeightsLoaded } from '../../src/inference/pipelines/text/lifecycle.js';

// A declared vision tower is not an executable one. Qwen3-VL loads its weights
// and config, but the shared ViT path applies no position encoding and the
// merger skips ln_q, so the runtime must refuse instead of emitting image tokens.
assert.deepEqual(resolveVisionArchitectureSupport('gemma4'), { architecture: 'gemma4', declared: true, executable: true, reason: null });
assert.deepEqual(resolveVisionArchitectureSupport('glmocr'), { architecture: 'glmocr', declared: true, executable: true, reason: null });
const qwen = resolveVisionArchitectureSupport(' qwen3vl ');
assert.equal(qwen.architecture, 'qwen3vl');
assert.equal(qwen.declared, true);
assert.equal(qwen.executable, false);
assert.match(qwen.reason, /rotary/);
assert.match(qwen.reason, /ln_q/);
assert.match(qwen.reason, /deepstack/);
assert.equal(resolveVisionArchitectureSupport('siglip').declared, false);
assert.equal(resolveVisionArchitectureSupport(undefined).executable, false);
assert.deepEqual(executableVisionArchitectures(), ['gemma4', 'glmocr']);

await assert.rejects(
  () => encodeImage({ pixels: new Uint8Array(4), width: 1, height: 1, visionConfig: { visionArchitecture: 'qwen3vl' }, weights: {} }),
  /Vision architecture "qwen3vl" cannot run: .*ln_q/
);
await assert.rejects(
  () => encodeImage({ pixels: new Uint8Array(4), width: 1, height: 1, visionConfig: { visionArchitecture: 'siglip' }, weights: {} }),
  /Unsupported vision architecture "siglip". Executable: gemma4, glmocr\./
);

// Pipeline entry points name the real reason when the manifest declares a
// tower the runtime cannot execute, and keep the manifest message otherwise.
const unsupported = { visionCapable: false, visionUnsupportedReason: 'vision architecture "qwen3vl" is declared by the manifest but not executable (no position encoding)' };
assert.equal(
  describeVisionUnavailable(unsupported, 'image transcription'),
  'Pipeline cannot run image transcription: vision architecture "qwen3vl" is declared by the manifest but not executable (no position encoding).'
);
assert.equal(describeVisionUnavailable({ visionCapable: false }, 'image embedding'), 'Pipeline does not support image embedding (no image_token_id in manifest).');
await assert.rejects(() => _ensureVisionWeightsLoaded.call(unsupported), /cannot run vision weights: vision architecture "qwen3vl"/);
await assert.rejects(() => _ensureVisionWeightsLoaded.call({ visionCapable: false }), /no image_token_id in manifest/);
console.log('vision architecture support ok');
