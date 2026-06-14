import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { createManifest } = await import('../../src/converter/core.js');
const { DEFAULT_MANIFEST_INFERENCE } = await import('../../src/config/schema/index.js');

const BASE_ARCH = {
  numLayers: 1,
  hiddenSize: 16,
  intermediateSize: 32,
  numAttentionHeads: 4,
  numKeyValueHeads: 4,
  headDim: 4,
  vocabSize: 128,
  maxSeqLen: 256,
};

const SHARDS = [
  {
    index: 0,
    filename: 'shard_00000.bin',
    size: 64,
    hash: 'a'.repeat(64),
    offset: 0,
  },
];

const MOE_TENSOR_LOCATIONS = {
  'model.layers.0.mlp.experts.gate_up_proj_blocks': {
    shard: 0,
    offset: 0,
    size: 2,
    shape: [1],
    dtype: 'F16',
    role: 'expert',
  },
};

const DENSE_TENSOR_LOCATIONS = {
  'model.layers.0.self_attn.q_proj.weight': {
    shard: 0,
    offset: 0,
    size: 2,
    shape: [1],
    dtype: 'F16',
    role: 'layer',
  },
};

const BASE_MANIFEST_OPTIONS = {
  source: 'test',
  modelType: 'gpt-oss',
  quantization: 'F16',
  hashAlgorithm: 'sha256',
  architecture: BASE_ARCH,
  inference: DEFAULT_MANIFEST_INFERENCE,
};

const DIFFUSION_GEMMA_GENERATION_CONFIG = {
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

{
  const manifest = createManifest(
    'gpt-oss-moe-test',
    {
      name: 'gpt-oss-moe-test',
      modelId: 'gpt-oss-moe-test',
      quantization: 'F16',
      tensors: [{ name: 'model.layers.0.mlp.experts.gate_up_proj_blocks', shape: [1], dtype: 'F16', size: 2 }],
      config: {
        model_type: 'gpt_oss',
        num_local_experts: 32,
        num_experts_per_tok: 4,
        eos_token_id: 1,
      },
      architecture: BASE_ARCH,
    },
    SHARDS,
    MOE_TENSOR_LOCATIONS,
    BASE_MANIFEST_OPTIONS
  );

  assert.deepEqual(
    manifest.moeConfig,
    { numExperts: 32, numExpertsPerToken: 4, expertFormat: 'gpt-oss' },
    'manifest should include derived GPT-OSS moeConfig'
  );
}

{
  const manifest = createManifest(
    'gemma4-moe-test',
    {
      name: 'gemma4-moe-test',
      modelId: 'gemma4-moe-test',
      quantization: 'F16',
      tensors: [{ name: 'model.layers.0.mlp.experts.gate_up_proj_blocks', shape: [1], dtype: 'F16', size: 2 }],
      config: {
        model_type: 'gemma4',
        text_config: {
          model_type: 'gemma4_text',
          num_experts: 128,
          top_k_experts: 8,
          eos_token_id: 1,
        },
      },
      architecture: BASE_ARCH,
    },
    SHARDS,
    MOE_TENSOR_LOCATIONS,
    {
      ...BASE_MANIFEST_OPTIONS,
      modelType: 'transformer',
    }
  );

  assert.deepEqual(
    manifest.moeConfig,
    { numExperts: 128, numExpertsPerToken: 8, expertFormat: 'mixtral' },
    'manifest should derive Gemma 4 moeConfig from top_k_experts'
  );
}

{
  const manifest = createManifest(
    'diffusiongemma-26b-a4b-test',
    {
      name: 'diffusiongemma-26b-a4b-test',
      modelId: 'diffusiongemma-26b-a4b-test',
      quantization: 'F16',
      tensors: [
        {
          name: 'model.decoder.layers.0.experts.gate_up_proj',
          shape: [128, 16, 4],
          dtype: 'F16',
          size: 128 * 16 * 4 * 2,
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
          moe_intermediate_size: 4,
          eos_token_id: 1,
        },
      },
      generationConfig: DIFFUSION_GEMMA_GENERATION_CONFIG,
      architecture: BASE_ARCH,
    },
    SHARDS,
    MOE_TENSOR_LOCATIONS,
    {
      ...BASE_MANIFEST_OPTIONS,
      modelType: 'diffusion_gemma',
    }
  );

  assert.equal(manifest.modelType, 'diffusion_gemma');
  assert.deepEqual(
    manifest.moeConfig,
    { numExperts: 128, numExpertsPerToken: 8, expertFormat: 'gemma4', expertIntermediateSize: 4 },
    'manifest should derive DiffusionGemma MoE contract with Gemma 4 expert format'
  );
  assert.deepEqual(
    manifest.inference.diffusionGemma,
    {
      canvasLength: 256,
      maxDenoisingSteps: 48,
      maxNewTokens: 256,
      tMin: 0.4,
      tMax: 0.8,
      entropyBound: 0.1,
      confidenceThreshold: 0.005,
      stabilityThreshold: 1,
      padTokenId: 0,
      eosTokenIds: [1, 106, 50],
      boiTokenId: 255999,
      eoiTokenId: 258882,
      imageTokenId: 258880,
      selfConditioning: true,
      decoderCacheMode: 'encoder_kv_readonly_canvas_concat',
      router: {
        scaleHiddenStates: true,
        normalizeTopK: true,
        perExpertScale: true,
      },
    },
    'manifest should stamp DiffusionGemma block-diffusion inference contract'
  );
}

{
  assert.throws(
    () => createManifest(
      'gpt-oss-moe-missing-topk',
      {
        name: 'gpt-oss-moe-missing-topk',
        modelId: 'gpt-oss-moe-missing-topk',
        quantization: 'F16',
        tensors: [{ name: 'model.layers.0.mlp.experts.gate_up_proj_blocks', shape: [1], dtype: 'F16', size: 2 }],
        config: {
          model_type: 'gpt_oss',
          num_local_experts: 32,
          eos_token_id: 1,
        },
        architecture: BASE_ARCH,
      },
      SHARDS,
      MOE_TENSOR_LOCATIONS,
      BASE_MANIFEST_OPTIONS
    ),
    /missing experts-per-token config/i,
    'converter should fail fast for MoE manifest missing top-k config'
  );
}

{
  const manifest = createManifest(
    'dense-test',
    {
      name: 'dense-test',
      modelId: 'dense-test',
      quantization: 'F16',
      tensors: [{ name: 'model.layers.0.self_attn.q_proj.weight', shape: [1], dtype: 'F16', size: 2 }],
      config: {
        model_type: 'llama',
        eos_token_id: 1,
      },
      architecture: BASE_ARCH,
    },
    SHARDS,
    DENSE_TENSOR_LOCATIONS,
    {
      ...BASE_MANIFEST_OPTIONS,
      modelType: 'llama',
    }
  );

  assert.equal(manifest.moeConfig, null, 'dense manifest should not include moeConfig');
}

console.log('core-moe-manifest.test: ok');
