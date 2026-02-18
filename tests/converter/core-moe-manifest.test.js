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
