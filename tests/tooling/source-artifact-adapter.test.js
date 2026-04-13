import assert from 'node:assert/strict';

const {
  assertDirectSourceRuntimeSupportedKind,
  resolveDirectSourceRuntimePlan,
  resolveSourceRuntimeBundleFromParsedArtifact,
} = await import('../../src/tooling/source-artifact-adapter.js');

const parsedArtifact = {
  sourceKind: 'gguf',
  config: {
    model_type: 'gemma4_text',
    architectures: ['Gemma4ForCausalLM'],
  },
  tensors: [
    {
      name: 'model.embed_tokens.weight',
      shape: [2, 2],
      dtype: 'F16',
      size: 8,
      offset: 0,
      sourcePath: 'model.gguf',
    },
    {
      name: 'model.layers.0.self_attn.q_proj.weight',
      shape: [2, 2],
      dtype: 'Q4_K_M',
      size: 144,
      offset: 8,
      sourcePath: 'model.gguf',
      layout: 'row',
    },
  ],
  architectureHint: 'gemma4',
  architecture: {
    numLayers: 1,
    hiddenSize: 2,
    intermediateSize: 4,
    numAttentionHeads: 1,
    numKeyValueHeads: 1,
    headDim: 2,
    vocabSize: 2,
    maxSeqLen: 8,
  },
  sourceQuantization: 'Q4_K_M',
  sourceFiles: [
    { path: 'model.gguf', size: 1024 },
  ],
  auxiliaryFiles: [],
  sourcePathForModelId: 'model.gguf',
};

const plan = resolveDirectSourceRuntimePlan({
  parsedArtifact,
  sourceQuantization: parsedArtifact.sourceQuantization,
  modelKind: 'transformer',
});

assert.equal(plan.modelType, 'gemma4_text');
assert.equal(plan.quantizationInfo.weights, 'q4k');
assert.equal(plan.quantizationInfo.embeddings, 'f16');
assert.equal(plan.quantizationInfo.layout, 'row');
assert.equal(plan.quantizationInfo.compute, 'f32');
assert.equal(plan.manifestQuantization, 'Q4_K_M');
assert.equal(plan.manifestInference.chatTemplate.type, 'gemma4');
assert.equal(plan.manifestConfig.hashAlgorithm, 'sha256');
assert.equal(assertDirectSourceRuntimeSupportedKind('litert-task'), 'litert-task');
assert.equal(assertDirectSourceRuntimeSupportedKind('litertlm'), 'litertlm');

await assert.rejects(
  () => resolveSourceRuntimeBundleFromParsedArtifact({
    parsedArtifact,
    runtimeLabel: 'source-artifact-adapter test',
    quantization: { weights: 'q4k' },
    async hashFileEntries(entries, hashAlgorithm) {
      return (entries ?? []).map((entry) => ({
        ...entry,
        hash: '0'.repeat(64),
        hashAlgorithm,
      }));
    },
  }),
  /converter-style quantization overrides are not supported/
);

console.log('source-artifact-adapter.test: ok');
