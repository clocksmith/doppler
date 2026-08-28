import assert from 'node:assert/strict';
import fs from 'node:fs';

import { expandExecutionV1 } from '../../src/config/schema/execution-v1.schema.js';
import { extractArchitecture } from '../../src/converter/artifact-identity.js';
import { applySourceTensorRules } from '../../src/converter/source-tensor-rules.js';

const configPath = 'src/config/conversion/glmocr/glm-ocr-f16-af32.json';
const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));

assert.equal(config.modelType, 'transformer');
assert.equal(config.output?.modelBaseId, 'glm-ocr-f16-af32');
assert.equal(config.output?.textOnly, false);
assert.deepEqual(
  {
    weights: config.quantization?.weights,
    embeddings: config.quantization?.embeddings,
    lmHead: config.quantization?.lmHead,
    vision: config.quantization?.vision,
    projector: config.quantization?.projector,
    computePrecision: config.quantization?.computePrecision,
  },
  {
    weights: 'f16',
    embeddings: 'f16',
    lmHead: 'f16',
    vision: 'f16',
    projector: 'f16',
    computePrecision: 'f32',
  }
);

assert.equal(
  config.manifest?.artifactIdentity?.sourceCheckpointId,
  'zai-org/GLM-OCR@ca5d8b3e287e52589e37c28385d9655ee4372f9d'
);
assert.deepEqual(config.manifest?.eosTokenId, [59246, 59253]);
assert.equal(config.manifest?.visionConfig?.vision_architecture, 'glmocr');
assert.equal(config.manifest?.visionConfig?.default_output_length, 6144);

assert.equal(config.inference?.attention?.queryPreAttnScalar, 128);
assert.equal(config.inference?.normalization?.rmsNormEps, 1e-5);
assert.equal(config.inference?.normalization?.rmsNormWeightOffset, false);
assert.deepEqual(config.inference?.rope?.mropeSection, [16, 24, 24]);
assert.equal(config.inference?.rope?.ropeInterleaved, true);
assert.equal(config.inference?.rope?.mropeInterleaved, true);
assert.equal(config.inference?.rope?.ropeFrequencyBaseDim, 128);
assert.equal(config.inference?.output?.tieWordEmbeddings, false);
assert.equal(config.inference?.chatTemplate?.type, 'glmocr');
assert.equal(config.inference?.chatTemplate?.thinking, null);

const expandedExecution = expandExecutionV1(config.execution);
assert.ok(expandedExecution.length > 0);
assert.deepEqual(
  config.execution.decode.find((step) => step[0] === 'ffn'),
  ['ffn', 'silu']
);
assert.deepEqual(
  config.execution.prefill.find((step) => step[0] === 'attention'),
  ['attention', 'attn_head128']
);
assert.deepEqual(
  config.execution.kernels.attn_head128,
  {
    kernel: 'attention_head128_f16kv.wgsl',
    entry: 'main',
    digest: 'sha256:559a3da4f3f5dcc4671f5a9b429a84209340a66a6e17f583bb329b2e23832c9b',
    precision: {
      kvDtype: 'f16',
    },
  }
);
assert.doesNotMatch(JSON.stringify(config.execution), /q4_/);

const sourceArchitecture = extractArchitecture({
  model_type: 'glm_ocr',
  text_config: {
    model_type: 'glm_ocr_text',
    vocab_size: 59392,
    hidden_size: 1536,
    intermediate_size: 4608,
    max_position_embeddings: 131072,
    num_attention_heads: 16,
    num_key_value_heads: 8,
    num_hidden_layers: 16,
    head_dim: 128,
    rope_parameters: {
      rope_theta: 10000,
    },
  },
}, null);
assert.deepEqual(
  {
    numLayers: sourceArchitecture.numLayers,
    hiddenSize: sourceArchitecture.hiddenSize,
    intermediateSize: sourceArchitecture.intermediateSize,
    numAttentionHeads: sourceArchitecture.numAttentionHeads,
    numKeyValueHeads: sourceArchitecture.numKeyValueHeads,
    headDim: sourceArchitecture.headDim,
    vocabSize: sourceArchitecture.vocabSize,
    maxSeqLen: sourceArchitecture.maxSeqLen,
    ropeTheta: sourceArchitecture.ropeTheta,
  },
  {
    numLayers: 16,
    hiddenSize: 1536,
    intermediateSize: 4608,
    numAttentionHeads: 16,
    numKeyValueHeads: 8,
    headDim: 128,
    vocabSize: 59392,
    maxSeqLen: 131072,
    ropeTheta: 10000,
  }
);

const auxiliarySuffixes = [
  'eh_proj.weight',
  'embed_tokens.weight',
  'enorm.weight',
  'hnorm.weight',
  'input_layernorm.weight',
  'mlp.down_proj.weight',
  'mlp.gate_up_proj.weight',
  'post_attention_layernorm.weight',
  'post_mlp_layernorm.weight',
  'post_self_attn_layernorm.weight',
  'self_attn.k_proj.weight',
  'self_attn.o_proj.weight',
  'self_attn.q_proj.weight',
  'self_attn.v_proj.weight',
  'shared_head.head.weight',
  'shared_head.norm.weight',
];
const selectedTensors = applySourceTensorRules([
  {
    name: 'model.language_model.layers.0.input_layernorm.weight',
    dtype: 'BF16',
    shape: [1536],
    offset: 0,
    size: 3072,
  },
  ...auxiliarySuffixes.map((suffix, index) => ({
    name: `model.language_model.layers.16.${suffix}`,
    dtype: 'BF16',
    shape: [1],
    offset: 3072 + index * 2,
    size: 2,
  })),
], config.sourceTensors);
assert.deepEqual(
  selectedTensors.map((tensor) => tensor.name),
  ['model.language_model.layers.0.input_layernorm.weight']
);

console.log('glmocr-conversion-config-contract.test: ok');
