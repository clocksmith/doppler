import assert from 'node:assert/strict';

const { parseTransformerModel } = await import('../../src/converter/parsers/transformer.js');

const jsonFiles = {
  'config.json': {
    architectures: ['EmbeddingGemmaModel'],
    model_type: 'embeddinggemma',
  },
  'modules.json': [
    { idx: 0, name: '0', path: '', type: 'sentence_transformers.models.Transformer' },
    { idx: 1, name: '1', path: '1_Pooling', type: 'sentence_transformers.models.Pooling' },
    { idx: 2, name: '2', path: '2_Dense', type: 'sentence_transformers.models.Dense' },
    { idx: 3, name: '3', path: '3_Dense', type: 'sentence_transformers.models.Dense' },
    { idx: 4, name: '4', path: '4_Normalize', type: 'sentence_transformers.models.Normalize' },
  ],
  '1_Pooling/config.json': {
    pooling_mode_cls_token: false,
    pooling_mode_mean_tokens: true,
    pooling_mode_max_tokens: false,
    pooling_mode_mean_sqrt_len_tokens: false,
    pooling_mode_weightedmean_tokens: false,
    pooling_mode_lasttoken: false,
    include_prompt: true,
  },
  '2_Dense/config.json': {
    in_features: 768,
    out_features: 3072,
    bias: false,
    activation_function: 'torch.nn.modules.linear.Identity',
  },
  '3_Dense/config.json': {
    in_features: 3072,
    out_features: 768,
    bias: false,
    activation_function: 'torch.nn.modules.linear.Identity',
  },
};

const safetensors = {
  'model.safetensors': [
    { name: 'embed_tokens.weight', shape: [16, 8], dtype: 'F32', size: 512, offset: 0, sourcePath: 'model.safetensors' },
  ],
  '2_Dense/model.safetensors': [
    { name: 'linear.weight', shape: [3072, 768], dtype: 'F32', size: 3072 * 768 * 4, offset: 0, sourcePath: '2_Dense/model.safetensors' },
  ],
  '3_Dense/model.safetensors': [
    { name: 'linear.weight', shape: [768, 3072], dtype: 'F32', size: 768 * 3072 * 4, offset: 0, sourcePath: '3_Dense/model.safetensors' },
  ],
};

const parsed = await parseTransformerModel({
  async readJson(suffix) {
    return jsonFiles[suffix];
  },
  async fileExists(suffix) {
    return Object.hasOwn(jsonFiles, suffix) || Object.hasOwn(safetensors, suffix);
  },
  async loadSingleSafetensors(suffix) {
    return safetensors[suffix] ?? [];
  },
  async loadShardedSafetensors() {
    return [];
  },
});

assert.equal(parsed.embeddingPostprocessor?.poolingMode, 'mean');
assert.equal(parsed.embeddingPostprocessor?.includePrompt, true);
assert.equal(parsed.embeddingPostprocessor?.normalize, 'l2');
assert.deepEqual(
  parsed.embeddingPostprocessor?.projections.map((projection) => projection.weightTensor),
  [
    'embedding_postprocessor.projections.0.weight',
    'embedding_postprocessor.projections.1.weight',
  ]
);
assert.equal(
  parsed.tensors.some((tensor) =>
    tensor.name === 'embedding_postprocessor.projections.0.weight'
    && tensor.role === 'lm_head'
    && tensor.group === 'embedding_postprocessor'
  ),
  true
);
assert.equal(
  parsed.tensors.some((tensor) =>
    tensor.name === 'embedding_postprocessor.projections.1.weight'
    && tensor.role === 'lm_head'
    && tensor.group === 'embedding_postprocessor'
  ),
  true
);

console.log('transformer-parser-embedding-postprocessor.test: ok');
