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

const diffusionGemmaGenerationConfig = {
  max_denoising_steps: 48,
  max_new_tokens: 256,
  t_min: 0.4,
  t_max: 0.8,
  stability_threshold: 1,
  confidence_threshold: 0.005,
  sampler_config: {
    _cls_name: 'EntropyBoundSamplerConfig',
    entropy_bound: 0.1,
  },
  pad_token_id: 0,
  eos_token_id: [1, 106, 50],
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

{
  let capturedManifest = null;
  const diffusionIo = {
    async readTensorData(tensor) {
      return new ArrayBuffer(tensor.size);
    },
    async writeShard(_index, _data) {
      return 'hash';
    },
    async writeManifest(manifest) {
      capturedManifest = manifest;
    },
  };

  await convertModel({
    name: 'diffusiongemma-26b-a4b-test',
    modelId: 'diffusiongemma-26b-a4b-test',
    quantization: 'F16',
    tensors: [
      {
        name: 'model.decoder.layers.0.experts.gate_up_proj',
        shape: [128, 16, 4],
        dtype: 'F16',
        size: 128 * 16 * 4 * 2,
        offset: 0,
      },
      {
        name: 'model.decoder.layers.0.router.scale',
        shape: [4],
        dtype: 'F16',
        size: 8,
        offset: 1,
      },
      {
        name: 'model.decoder.layers.0.post_feedforward_layernorm_1.weight',
        shape: [4],
        dtype: 'F16',
        size: 8,
        offset: 2,
      },
      {
        name: 'model.encoder.language_model.layers.0.self_attn.q_proj.weight',
        shape: [4, 4],
        dtype: 'F16',
        size: 32,
        offset: 3,
      },
    ],
    config: {
      model_type: 'diffusion_gemma',
      canvas_length: 256,
      boi_token_id: 255999,
      eoi_token_id: 258882,
      image_token_id: 258880,
      text_config: {
        model_type: 'diffusion_gemma_text',
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
    generationConfig: diffusionGemmaGenerationConfig,
  }, diffusionIo, {
    ...baseOptions,
    modelId: 'diffusiongemma-26b-a4b-test',
    modelType: 'diffusion_gemma',
  });

  assert.ok(capturedManifest, 'DiffusionGemma conversion should emit a manifest');
  assert.equal(capturedManifest.modelType, 'diffusion_gemma');
  assert.deepEqual(
    capturedManifest.moeConfig,
    { numExperts: 128, numExpertsPerToken: 8, expertFormat: 'gemma4', expertIntermediateSize: 4 }
  );
  assert.equal(capturedManifest.inference.diffusionGemma.canvasLength, 256);
  assert.equal(capturedManifest.inference.diffusionGemma.padTokenId, 0);
  assert.deepEqual(capturedManifest.inference.diffusionGemma.eosTokenIds, [1, 106, 50]);
  assert.equal(
    capturedManifest.inference.diffusionGemma.decoderCacheMode,
    'encoder_kv_readonly_canvas_concat'
  );
}

console.log('core-gemma4-unsupported-architecture.test: ok');
