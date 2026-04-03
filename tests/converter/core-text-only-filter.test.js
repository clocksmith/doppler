import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { convertModel } = await import('../../src/converter/core.js');
const { createConverterConfig } = await import('../../src/config/schema/converter.schema.js');
const { DEFAULT_MANIFEST_INFERENCE } = await import('../../src/config/schema/index.js');

function toArrayBuffer(view) {
  return view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength);
}

const sourceTensor = new Float32Array([0.5, -1.0, 1.5, -2.0]);
const sourceBytes = new Uint8Array(sourceTensor.buffer);

const languageTensorName = 'language_model.model.embed_tokens.weight';
const visionTensorName = 'vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight';
const projectorTensorName = 'multi_modal_projector.mm_input_projection_weight';
const embedVisionTensorName = 'model.embed_vision.embedding_projection.weight';

const model = {
  name: 'text-only-filter-test',
  modelId: 'text-only-filter-test',
  quantization: 'F32',
  tensors: [
    {
      name: languageTensorName,
      shape: [2, 2],
      dtype: 'F32',
      size: 16,
      offset: 0,
    },
    {
      name: visionTensorName,
      shape: [2, 2],
      dtype: 'F32',
      size: 16,
      offset: 16,
    },
    {
      name: projectorTensorName,
      shape: [2, 2],
      dtype: 'F32',
      size: 16,
      offset: 32,
    },
    {
      name: embedVisionTensorName,
      shape: [2, 2],
      dtype: 'F32',
      size: 16,
      offset: 48,
    },
  ],
  config: {
    model_type: 'gemma3',
    text_config: {
      model_type: 'gemma3_text',
      num_hidden_layers: 1,
      hidden_size: 2,
      intermediate_size: 8,
      num_attention_heads: 1,
      num_key_value_heads: 1,
      head_dim: 2,
      vocab_size: 16,
      max_position_embeddings: 8,
      use_bidirectional_attention: false,
    },
  },
};

const readTensorNames = [];
let capturedManifest = null;
const io = {
  async readTensorData(tensor) {
    readTensorNames.push(tensor.name);
    return toArrayBuffer(sourceBytes);
  },
  async writeShard(_index, _data) {
    return 'hash';
  },
  async writeManifest(manifest) {
    capturedManifest = manifest;
  },
};

await convertModel(model, io, {
  modelId: 'text-only-filter-test',
  modelType: 'transformer',
  quantization: 'F16',
  quantizationInfo: {
    weights: 'f16',
    embeddings: 'f16',
    compute: 'f16',
    variantTag: 'f16',
  },
  architecture: {
    numLayers: 1,
    hiddenSize: 2,
    intermediateSize: 8,
    numAttentionHeads: 1,
    numKeyValueHeads: 1,
    headDim: 2,
    vocabSize: 16,
    maxSeqLen: 8,
    ropeTheta: 1000000,
  },
  inference: {
    ...DEFAULT_MANIFEST_INFERENCE,
  },
  eosTokenId: 1,
  converterConfig: createConverterConfig({
    output: {
      textOnly: true,
    },
  }),
});

assert.deepEqual(readTensorNames, [languageTensorName]);
assert.ok(capturedManifest, 'manifest should be written');
assert.ok(capturedManifest.tensors?.[languageTensorName], 'language tensor should be in manifest');
assert.equal(capturedManifest.tensors?.[visionTensorName], undefined);
assert.equal(capturedManifest.tensors?.[projectorTensorName], undefined);
assert.equal(capturedManifest.tensors?.[embedVisionTensorName], undefined);
assert.equal(Object.keys(capturedManifest.tensors ?? {}).length, 1);

console.log('core-text-only-filter.test: ok');
