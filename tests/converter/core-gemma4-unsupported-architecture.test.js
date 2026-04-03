import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { convertModel } = await import('../../src/converter/core.js');
const { createConverterConfig } = await import('../../src/config/schema/converter.schema.js');
const { DEFAULT_MANIFEST_INFERENCE } = await import('../../src/config/schema/index.js');

function toArrayBuffer(view) {
  return view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength);
}

function createIo() {
  return {
    async readTensorData(_tensor) {
      return toArrayBuffer(new Uint8Array([0, 0, 0, 0]));
    },
    async writeShard(_index, _data) {
      return 'hash';
    },
    async writeManifest(_manifest) {
      throw new Error('manifest write should not happen for unsupported Gemma 4 models');
    },
  };
}

const baseOptions = {
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
    hiddenSize: 4,
    intermediateSize: 8,
    numAttentionHeads: 1,
    numKeyValueHeads: 1,
    headDim: 4,
    vocabSize: 16,
    maxSeqLen: 32,
    ropeTheta: 1000000,
  },
  inference: DEFAULT_MANIFEST_INFERENCE,
  eosTokenId: 1,
  converterConfig: createConverterConfig({
    output: {
      textOnly: true,
    },
  }),
};

await assert.rejects(
  () => convertModel({
    name: 'gemma4-e2b-test',
    modelId: 'gemma4-e2b-test',
    quantization: 'F16',
    tensors: [
      {
        name: 'model.language_model.embed_tokens.weight',
        shape: [4, 4],
        dtype: 'F16',
        size: 16,
        offset: 0,
      },
      {
        name: 'model.language_model.embed_tokens_per_layer.weight',
        shape: [16, 8],
        dtype: 'F16',
        size: 16 * 8 * 2,
        offset: 1,
      },
    ],
    config: {
      model_type: 'gemma4',
      text_config: {
        model_type: 'gemma4_text',
        hidden_size_per_layer_input: 8,
        num_hidden_layers: 1,
        hidden_size: 4,
        intermediate_size: 8,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 4,
        vocab_size: 16,
        max_position_embeddings: 32,
      },
    },
  }, createIo(), {
    ...baseOptions,
    modelId: 'gemma4-e2b-test',
  }),
  /missing required per-layer input tensors: per_layer_input_gate.weight, per_layer_projection.weight, post_per_layer_input_norm.weight, per_layer_model_projection.weight, per_layer_projection_norm.weight/i,
  'Gemma 4 PLE models should fail fast when required per-layer input tensors are absent'
);

await assert.rejects(
  () => convertModel({
    name: 'gemma4-26b-test',
    modelId: 'gemma4-26b-test',
    quantization: 'F16',
    tensors: [
      {
        name: 'model.language_model.layers.0.experts.gate_up_proj',
        shape: [128, 16, 4],
        dtype: 'F16',
        size: 128 * 16 * 4 * 2,
        offset: 0,
      },
      {
        name: 'model.language_model.layers.0.router.scale',
        shape: [4],
        dtype: 'F16',
        size: 8,
        offset: 1,
      },
      {
        name: 'model.language_model.layers.0.post_feedforward_layernorm_1.weight',
        shape: [4],
        dtype: 'F16',
        size: 8,
        offset: 2,
      },
    ],
    config: {
      model_type: 'gemma4',
      text_config: {
        model_type: 'gemma4_text',
        enable_moe_block: true,
        num_experts: 128,
        top_k_experts: 8,
        num_hidden_layers: 1,
        hidden_size: 4,
        intermediate_size: 8,
        moe_intermediate_size: 4,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 4,
        vocab_size: 16,
        max_position_embeddings: 32,
      },
    },
  }, createIo(), {
    ...baseOptions,
    modelId: 'gemma4-26b-test',
  }),
  /Gemma 4 MoE decoder blocks require Gemma-specific router scaling and dual dense\+MoE FFN execution/i,
  'Gemma 4 MoE models should fail fast during conversion'
);

console.log('core-gemma4-unsupported-architecture.test: ok');
