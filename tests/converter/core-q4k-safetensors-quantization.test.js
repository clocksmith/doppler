import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { convertModel } = await import('../../src/converter/core.js');
const { createConverterConfig } = await import('../../src/config/schema/converter.schema.js');
const { float32ToFloat16 } = await import('../../src/converter/quantizer.js');
const { DEFAULT_MANIFEST_INFERENCE } = await import('../../src/config/schema/index.js');

function toArrayBuffer(view) {
  return view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength);
}

function buildF16Tensor(length, base = 0) {
  const out = new Uint16Array(length);
  for (let i = 0; i < length; i++) {
    out[i] = float32ToFloat16(base + (i % 17) * 0.125);
  }
  return out;
}

const qProjShape = [4, 256]; // 1024 elems => eligible for quantization.
const normShape = [256]; // 1D norm should never be q4k-quantized.

const tensors = [
  {
    name: 'model.layers.0.self_attn.q_proj.weight',
    shape: qProjShape,
    dtype: 'F16',
    size: qProjShape[0] * qProjShape[1] * 2,
    offset: 0,
  },
  {
    name: 'model.layers.0.input_layernorm.weight',
    shape: normShape,
    dtype: 'F16',
    size: normShape[0] * 2,
    offset: qProjShape[0] * qProjShape[1] * 2,
  },
];

const tensorData = new Map([
  ['model.layers.0.self_attn.q_proj.weight', buildF16Tensor(qProjShape[0] * qProjShape[1], -1.0)],
  ['model.layers.0.input_layernorm.weight', buildF16Tensor(normShape[0], 0.5)],
]);

const model = {
  name: 'q4k-safetensors-test',
  modelId: 'q4k-safetensors-test',
  quantization: 'F16',
  tensors,
  config: {
    model_type: 'gemma3',
    architectures: ['Gemma3ForCausalLM'],
    num_hidden_layers: 1,
    hidden_size: 256,
    intermediate_size: 1024,
    num_attention_heads: 4,
    num_key_value_heads: 4,
    head_dim: 64,
    vocab_size: 256,
    max_position_embeddings: 128,
  },
};

let capturedManifest = null;
let capturedShards = [];
const io = {
  async readTensorData(tensor) {
    const values = tensorData.get(tensor.name);
    if (!values) {
      throw new Error(`Missing tensor data for ${tensor.name}`);
    }
    return toArrayBuffer(new Uint8Array(values.buffer));
  },
  async writeShard(index, data) {
    capturedShards[index] = new Uint8Array(data);
    return 'hash';
  },
  async writeManifest(manifest) {
    capturedManifest = manifest;
  },
};

await convertModel(model, io, {
  modelId: 'q4k-safetensors-test',
  modelType: 'transformer',
  quantization: 'Q4_K_M',
  quantizationInfo: {
    weights: 'q4k',
    embeddings: 'f16',
    compute: 'f16',
    layout: 'row',
    variantTag: 'wq4k-ef16',
  },
  architecture: {
    numLayers: 1,
    hiddenSize: 256,
    intermediateSize: 1024,
    numAttentionHeads: 4,
    numKeyValueHeads: 4,
    headDim: 64,
    vocabSize: 256,
    maxSeqLen: 128,
    ropeTheta: 1000000,
  },
  inference: {
    ...DEFAULT_MANIFEST_INFERENCE,
    presetId: 'gemma3',
  },
  eosTokenId: 1,
  converterConfig: createConverterConfig(),
});

assert.ok(capturedManifest, 'manifest should be written');
assert.equal(capturedManifest.quantization, 'Q4_K_M');
assert.equal(capturedManifest.quantizationInfo?.weights, 'q4k');

const qProjLoc = capturedManifest.tensors?.['model.layers.0.self_attn.q_proj.weight'];
const normLoc = capturedManifest.tensors?.['model.layers.0.input_layernorm.weight'];
assert.ok(qProjLoc, 'q_proj tensor location missing');
assert.ok(normLoc, 'norm tensor location missing');

// 4 rows * ceil(256/256) blocks per row * 144 bytes per block
const expectedQProjBytes = 4 * 1 * 144;
assert.equal(qProjLoc.dtype, 'Q4_K_M', 'q_proj should be stored as Q4_K_M');
assert.equal(qProjLoc.layout, 'row', 'q_proj should record row layout');
assert.equal(qProjLoc.size, expectedQProjBytes, 'q_proj should be physically q4k-packed');

assert.equal(normLoc.dtype, 'F16', 'norm should remain F16');
assert.equal(normLoc.size, 256 * 2, 'norm byte size should remain F16-sized');

const shard0 = capturedShards[0];
assert.ok(shard0 instanceof Uint8Array, 'first shard should be captured');
assert.equal(
  shard0.byteLength,
  expectedQProjBytes + normLoc.size,
  'shard bytes should reflect packed q4k tensor + unquantized norm tensor'
);

console.log('core-q4k-safetensors-quantization.test: ok');
