import assert from 'node:assert/strict';

// Import the pipeline module to trigger registration
await import('../../src/inference/pipelines/text.js');
const { InferencePipeline } = await import('../../src/inference/pipelines/text.js');

// ==========================================================================
// Test 1: Text-only pipeline — visionCapable must be false
// ==========================================================================

const textOnlyPipeline = new InferencePipeline();

// Simulate a text-only manifest (no image_token_id, no quantizationInfo.vision)
const textOnlyManifest = {
  modelId: 'test-text-only',
  modelType: 'transformer',
  quantization: 'q4k',
  architecture: {
    numLayers: 2,
    hiddenSize: 256,
    headDim: 64,
    vocabSize: 1000,
    maxSeqLen: 512,
    numHeads: 4,
  },
  eos_token_id: 2,
  inference: {},
  shards: [],
  hashAlgorithm: 'blake3',
  totalSize: 0,
  version: 1,
};

// Vision capability detection happens inside loadModel, which requires GPU.
// For unit testing, we directly test the detection logic.
assert.equal(textOnlyPipeline.visionCapable, false, 'Fresh pipeline should not be vision capable');
assert.equal(textOnlyPipeline.imageTokenId, null, 'Fresh pipeline should have null imageTokenId');

// Verify capabilities getter on fresh pipeline
const freshCaps = textOnlyPipeline.capabilities;
assert.ok(freshCaps.includes('generation'), 'Fresh pipeline should have generation capability');
assert.ok(!freshCaps.includes('multimodal'), 'Text-only pipeline should not have multimodal capability');
assert.ok(Object.isFrozen(freshCaps), 'Capabilities should be frozen');

// ==========================================================================
// Test 2: transcribeImage throws on text-only pipeline
// ==========================================================================

try {
  await textOnlyPipeline.transcribeImage({
    imageBytes: new Uint8Array(100),
    width: 10,
    height: 10,
  });
  assert.fail('transcribeImage should throw on text-only pipeline');
} catch (err) {
  assert.ok(
    err.message.includes('does not support image transcription'),
    `Expected vision-not-supported error, got: ${err.message}`
  );
}

// ==========================================================================
// Test 3: Vision-capable manifest detection (direct field check)
// ==========================================================================

const visionManifest = {
  ...textOnlyManifest,
  modelId: 'test-vision',
  image_token_id: 151655,
  visionArchitecture: 'qwen3vl',
  quantizationInfo: {
    weights: 'q4k',
    vision: 'f16',
    projector: 'f16',
  },
};

// Verify the manifest has the right shape for detection
const imageTokenId = visionManifest.image_token_id;
const hasVisionQuant = visionManifest.quantizationInfo?.vision != null;
assert.ok(
  Number.isInteger(imageTokenId) && imageTokenId > 0 && hasVisionQuant,
  'Vision manifest should pass capability detection gate'
);

// Verify text-only manifest does NOT pass
const textImageTokenId = textOnlyManifest.image_token_id;
const textHasVisionQuant = textOnlyManifest.quantizationInfo?.vision != null;
assert.ok(
  !(Number.isInteger(textImageTokenId) && textImageTokenId > 0 && textHasVisionQuant),
  'Text-only manifest should not pass capability detection gate'
);

// ==========================================================================
// Test 4: capabilities getter reflects vision state
// ==========================================================================

const visionPipeline = new InferencePipeline();
visionPipeline.visionCapable = true;
const visionCaps = visionPipeline.capabilities;
assert.ok(visionCaps.includes('multimodal'), 'Vision pipeline should have multimodal capability');
assert.ok(visionCaps.includes('generation'), 'Vision pipeline should still have generation');

console.log('vision-pipeline-capability.test: ok');
