import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { convertModel } = await import('../../src/converter/core.js');
const { createConverterConfig } = await import('../../src/config/schema/converter.schema.js');
const { float16ToFloat32 } = await import('../../src/converter/quantizer.js');
const { DEFAULT_MANIFEST_INFERENCE } = await import('../../src/config/schema/index.js');

function toArrayBuffer(view) {
  return view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength);
}

const BF16_ROUND_VIEW = new DataView(new ArrayBuffer(4));

function float32ToBFloat16(value) {
  BF16_ROUND_VIEW.setFloat32(0, value, true);
  const bits = BF16_ROUND_VIEW.getUint32(0, true);
  const lsb = (bits >> 16) & 1;
  const roundingBias = 0x7fff + lsb;
  return ((bits + roundingBias) >> 16) & 0xffff;
}

const sourceValues = new Float32Array([1.25, -2.5, 3.75, -4.0]);
const bf16Data = new Uint16Array(sourceValues.length);
for (let i = 0; i < sourceValues.length; i++) {
  bf16Data[i] = float32ToBFloat16(sourceValues[i]);
}

const tensors = [
  {
    name: 'model.layers.0.self_attn.q_proj.weight',
    shape: [2, 2],
    dtype: 'BF16',
    size: 8,
    offset: 0,
  },
];

const model = {
  name: 'bf16-f16-conversion-test',
  modelId: 'bf16-f16-conversion-test',
  quantization: 'F16',
  tensors,
  config: {
    model_type: 'gemma3_text',
    architectures: ['Gemma3ForCausalLM'],
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
};

let capturedManifest = null;
let capturedShard = null;
const io = {
  async readTensorData(_tensor) {
    return toArrayBuffer(new Uint8Array(bf16Data.buffer));
  },
  async writeShard(_index, data) {
    capturedShard = new Uint8Array(data);
    return 'hash';
  },
  async writeManifest(manifest) {
    capturedManifest = manifest;
  },
};

await convertModel(model, io, {
  modelId: 'bf16-f16-conversion-test',
  modelType: 'transformer',
  quantization: 'F16',
  quantizationInfo: {
    weights: 'f16',
    embeddings: 'f16',
    compute: 'f16',
    variantTag: 'wf16',
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
    presetId: 'gemma3',
  },
  eosTokenId: 1,
  converterConfig: createConverterConfig(),
});

assert.ok(capturedManifest, 'manifest should be written');
assert.ok(capturedShard, 'converted shard should be written');

const loc = capturedManifest.tensors?.['model.layers.0.self_attn.q_proj.weight'];
assert.ok(loc, 'tensor location missing');
assert.equal(loc.dtype, 'F16', 'BF16 source should be converted to F16 for wf16 output');
assert.equal(loc.size, 8, 'tensor should remain 2 bytes/element after conversion');

const outF16 = new Uint16Array(
  capturedShard.buffer,
  capturedShard.byteOffset + loc.offset,
  sourceValues.length
);
const outF32 = Array.from(outF16, (h) => float16ToFloat32(h));
for (let i = 0; i < sourceValues.length; i++) {
  assert.ok(
    Math.abs(outF32[i] - sourceValues[i]) < 1e-2,
    `converted value mismatch at ${i}: got ${outF32[i]}, expected ${sourceValues[i]}`
  );
}

console.log('core-bf16-f16-conversion.test: ok');
