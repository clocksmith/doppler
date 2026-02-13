import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { convertModel } = await import('../../src/converter/core.js');
const { createConverterConfig } = await import('../../src/config/schema/converter.schema.js');
const { DEFAULT_MANIFEST_INFERENCE } = await import('../../src/config/schema/index.js');

function toArrayBuffer(view) {
  return view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength);
}

const tensors = [
  {
    name: 'embed_tokens.weight',
    shape: [2, 2],
    dtype: 'F32',
    size: 16,
    offset: 0,
  },
  {
    name: 'model.layers.0.self_attn.q_proj.weight',
    shape: [2, 2],
    dtype: 'F32',
    size: 16,
    offset: 16,
  },
];

const tensorData = new Map([
  ['embed_tokens.weight', new Float32Array([1.0, 2.0, 3.0, 4.0])],
  ['model.layers.0.self_attn.q_proj.weight', new Float32Array([5.0, 6.0, 7.0, 8.0])],
]);

const model = {
  name: 'mixed-quant-test',
  modelId: 'mixed-quant-test',
  quantization: 'F16',
  tensors,
  config: {
    model_type: 'embeddinggemma',
    architectures: ['Gemma3TextModel'],
    num_hidden_layers: 1,
    hidden_size: 2,
    intermediate_size: 2,
    num_attention_heads: 1,
    num_key_value_heads: 1,
    head_dim: 2,
    vocab_size: 2,
    max_position_embeddings: 8,
  },
};

let capturedManifest = null;
const io = {
  async readTensorData(tensor) {
    const values = tensorData.get(tensor.name);
    if (!values) {
      throw new Error(`Missing tensor data for ${tensor.name}`);
    }
    return toArrayBuffer(new Uint8Array(values.buffer));
  },
  async writeShard(_index, _data) {
    return 'hash';
  },
  async writeManifest(manifest) {
    capturedManifest = manifest;
  },
};

await convertModel(model, io, {
  modelId: 'mixed-quant-test',
  modelType: 'embedding',
  quantization: 'F16',
  quantizationInfo: {
    weights: 'f16',
    embeddings: 'f32',
    compute: 'f16',
    variantTag: 'wf16-ef32',
  },
  architecture: {
    numLayers: 1,
    hiddenSize: 2,
    intermediateSize: 2,
    numAttentionHeads: 1,
    numKeyValueHeads: 1,
    headDim: 2,
    vocabSize: 2,
    maxSeqLen: 8,
    ropeTheta: 1000000,
  },
  inference: {
    ...DEFAULT_MANIFEST_INFERENCE,
    presetId: 'embeddinggemma',
    output: {
      ...DEFAULT_MANIFEST_INFERENCE.output,
      tieWordEmbeddings: false,
      scaleEmbeddings: true,
    },
  },
  eosTokenId: 1,
  converterConfig: createConverterConfig(),
});

assert.ok(capturedManifest, 'manifest should be written');
assert.equal(capturedManifest.quantization, 'F16');
assert.equal(capturedManifest.quantizationInfo?.embeddings, 'f32');

const embedLoc = capturedManifest.tensors?.['embed_tokens.weight'];
const qProjLoc = capturedManifest.tensors?.['model.layers.0.self_attn.q_proj.weight'];
assert.ok(embedLoc, 'embed tensor location missing');
assert.ok(qProjLoc, 'q_proj tensor location missing');

assert.equal(embedLoc.dtype, 'F32', 'embedding tensor should remain F32');
assert.equal(qProjLoc.dtype, 'F16', 'non-embedding tensor should downcast to F16');
assert.equal(embedLoc.size, 16, 'embedding tensor byte size should stay F32-sized');
assert.equal(qProjLoc.size, 8, 'q_proj tensor byte size should become F16-sized');

console.log('core-mixed-quantization.test: ok');
