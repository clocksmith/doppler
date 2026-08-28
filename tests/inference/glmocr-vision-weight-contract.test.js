import assert from 'node:assert/strict';

import { _loadVisionWeights } from '../../src/inference/pipelines/text/lifecycle.js';

const calls = [];
const loader = {
  async loadTensor(name) {
    calls.push(name);
    return { label: name };
  },
  async loadGpuTensor(name) {
    calls.push(name);
    return {
      buffer: { label: `${name}.buffer` },
      dtype: 'f16',
      shape: [1],
      label: name,
    };
  },
};
const pipeline = {
  dopplerLoader: loader,
  runtimeConfig: { loading: {} },
  visionConfig: {
    visionArchitecture: 'glmocr',
    depth: 2,
  },
  modelConfig: {
    hiddenSize: 1536,
  },
};

await _loadVisionWeights.call(pipeline);

assert.equal(pipeline.visionWeights.textHiddenSize, 1536);
assert.equal(pipeline.visionWeights.layers.length, 2);
assert.equal(
  pipeline.visionWeights.layers[0].qNormWeight.label,
  'model.visual.blocks.0.attn.q_norm.weight'
);
assert.equal(pipeline.visionWeights.layers[0].qNormWeight.dtype, 'f16');
assert.equal(
  pipeline.visionWeights.merger.postProjectionNormBias.label,
  'model.visual.merger.post_projection_norm.bias'
);
assert.ok(calls.includes('model.visual.patch_embed.proj.weight'));
assert.ok(calls.includes('model.visual.downsample.weight'));
assert.ok(calls.includes('model.visual.blocks.1.mlp.down_proj.bias'));
assert.equal(calls.length, 11 + (2 * 14));

console.log('glmocr-vision-weight-contract.test: ok');
