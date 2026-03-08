import assert from 'node:assert/strict';

const { parseModelConfig } = await import('../../src/inference/pipelines/text/config.js');
const { DEFAULT_MANIFEST_INFERENCE } = await import('../../src/config/schema/index.js');

const baseManifest = {
  version: 1,
  modelId: 'lfm2-runtime-intermediate-test',
  modelType: 'transformer',
  quantization: 'Q4_K_M',
  architecture: {
    numLayers: 2,
    hiddenSize: 2048,
    intermediateSize: 12288,
    numAttentionHeads: 32,
    numKeyValueHeads: 8,
    headDim: 64,
    vocabSize: 65536,
    maxSeqLen: 4096,
  },
  inference: {
    ...DEFAULT_MANIFEST_INFERENCE,
    presetId: 'lfm2',
  },
  eos_token_id: 7,
  tensors: {
    'model.layers.0.feed_forward.w1.weight': {
      shape: [8192, 2048],
      dtype: 'Q4_K_M',
      size: 1,
      shard: 0,
      offset: 0,
      role: 'matmul',
    },
    'model.layers.0.feed_forward.w2.weight': {
      shape: [2048, 8192],
      dtype: 'Q4_K_M',
      size: 1,
      shard: 0,
      offset: 1,
      role: 'matmul',
    },
    'model.layers.0.feed_forward.w3.weight': {
      shape: [8192, 2048],
      dtype: 'Q4_K_M',
      size: 1,
      shard: 0,
      offset: 2,
      role: 'matmul',
    },
  },
};

{
  assert.throws(
    () => parseModelConfig(baseManifest, {}),
    /FFN tensors imply 8192/
  );
}

{
  const nonLfm2 = {
    ...baseManifest,
    modelId: 'non-lfm2-runtime-test',
    inference: {
      ...baseManifest.inference,
      presetId: 'gemma3',
    },
  };
  const parsed = parseModelConfig(nonLfm2, {});
  assert.equal(parsed.intermediateSize, 12288);
}

{
  const linearNormFromArchitecture = {
    ...baseManifest,
    modelId: 'linear-norm-mode-arch-test',
    inference: {
      ...baseManifest.inference,
      presetId: 'gemma3',
    },
    architecture: {
      ...baseManifest.architecture,
      linearNumKeyHeads: 16,
      linearNumValueHeads: 16,
      linearKeyHeadDim: 128,
      linearValueHeadDim: 128,
      linearConvKernelDim: 4,
      linearNormMode: 'per_head',
    },
  };
  const parsed = parseModelConfig(linearNormFromArchitecture, {});
  assert.equal(parsed.linearNormMode, 'per_head');
}

{
  const linearNormFromConfig = {
    ...baseManifest,
    modelId: 'linear-norm-mode-config-test',
    inference: {
      ...baseManifest.inference,
      presetId: 'gemma3',
    },
    config: {
      model_type: 'qwen2',
      linear_norm_mode: 'shared',
    },
  };
  const parsed = parseModelConfig(linearNormFromConfig, {});
  assert.equal(parsed.linearNormMode, 'shared');
}

{
  const invalidLinearNormMode = {
    ...baseManifest,
    modelId: 'linear-norm-mode-invalid-test',
    inference: {
      ...baseManifest.inference,
      presetId: 'gemma3',
    },
    config: {
      model_type: 'qwen2',
      linear_norm_mode: 'bad-mode',
    },
  };
  assert.throws(
    () => parseModelConfig(invalidLinearNormMode, {}),
    /unsupported linear_norm_mode/i
  );
}

console.log('config-lfm2-intermediate-runtime.test: ok');
