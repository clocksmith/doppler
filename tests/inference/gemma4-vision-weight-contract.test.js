import assert from 'node:assert/strict';

import { _loadVisionWeights } from '../../src/inference/pipelines/text/lifecycle.js';

const table = {
  buffer: { label: 'position_table' },
  dtype: 'f16',
  shape: Object.freeze([2, 10240, 768]),
};
const calls = [];
const loader = {
  async loadTensor(name) {
    calls.push(['loadTensor', name]);
    return { label: name };
  },
  async loadGpuTensor(name) {
    calls.push(['loadGpuTensor', name]);
    return table;
  },
};
const pipeline = {
  dopplerLoader: loader,
  runtimeConfig: { loading: {} },
  visionConfig: {
    visionArchitecture: 'gemma4',
    depth: 0,
  },
  modelConfig: {
    hiddenSize: 2304,
  },
};

await _loadVisionWeights.call(pipeline);

assert.equal(pipeline.visionWeights.patchPositionEmbeddingTable, table);
assert.deepEqual(calls, [
  ['loadTensor', 'model.vision_tower.patch_embedder.input_proj.weight'],
  ['loadGpuTensor', 'model.vision_tower.patch_embedder.position_embedding_table'],
  ['loadTensor', 'model.embed_vision.embedding_projection.weight'],
]);

console.log('gemma4-vision-weight-contract.test: ok');
