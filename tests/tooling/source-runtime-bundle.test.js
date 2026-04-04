import assert from 'node:assert/strict';

const { DEFAULT_MANIFEST_INFERENCE } = await import('../../src/config/schema/index.js');
const {
  buildSourceRuntimeBundle,
  createSourceStorageContext,
} = await import('../../src/tooling/source-runtime-bundle.js');

const inference = JSON.parse(JSON.stringify(DEFAULT_MANIFEST_INFERENCE));

const tensors = [
  {
    name: 'model.embed_tokens.weight',
    shape: [4, 4],
    dtype: 'F16',
    size: 32,
    offset: 0,
    sourcePath: 'weights_a.safetensors',
  },
  {
    name: 'model.layers.0.self_attn.q_proj.weight',
    shape: [4, 4],
    dtype: 'F16',
    size: 32,
    offset: 32,
    sourcePath: 'weights_a.safetensors',
  },
  {
    name: 'lm_head.weight',
    shape: [4, 4],
    dtype: 'F16',
    size: 32,
    offset: 0,
    sourcePath: 'weights_b.safetensors',
  },
  {
    name: 'model.layers.0.self_attn.q_proj.input_max',
    shape: [],
    dtype: 'BF16',
    size: 2,
    offset: 32,
    sourcePath: 'weights_b.safetensors',
  },
];

const sourceFiles = [
  { path: 'weights_a.safetensors', size: 64 },
  { path: 'weights_b.safetensors', size: 34 },
];

const bundle = await buildSourceRuntimeBundle({
  modelId: 'source-runtime-unit-test',
  modelType: 'transformer',
  architecture: {
    numLayers: 1,
    hiddenSize: 4,
    intermediateSize: 8,
    numAttentionHeads: 1,
    numKeyValueHeads: 1,
    headDim: 4,
    vocabSize: 4,
    maxSeqLen: 8,
    ropeTheta: 10000,
  },
  architectureHint: 'gemma3',
  rawConfig: {
    model_type: 'gemma3_text',
    architectures: ['Gemma3ForCausalLM'],
    eos_token_id: 2,
  },
  inference,
  tensors,
  sourceFiles,
  sourceQuantization: 'f16',
  hashAlgorithm: 'sha256',
  tokenizerJson: {
    model: {
      vocab: {
        '<bos>': 0,
        '<eos>': 1,
        'hello': 2,
        'world': 3,
      },
    },
    special_tokens: {
      bos_token: '<bos>',
      eos_token: '<eos>',
    },
    added_tokens_decoder: {
      '1': { content: '<eos>' },
    },
  },
  tokenizerModelName: 'tokenizer.model',
});

assert.equal(bundle.manifest.modelId, 'source-runtime-unit-test');
assert.equal(bundle.manifest.shards.length, 2);
assert.equal(bundle.manifest.hashAlgorithm, 'sha256');
assert.equal(bundle.manifest.tensors['model.embed_tokens.weight'].group, 'embed');
assert.equal(bundle.manifest.tensors['model.layers.0.self_attn.q_proj.weight'].group, 'layer.0');
assert.equal(bundle.manifest.tensors['lm_head.weight'].group, 'head');
assert.deepEqual(bundle.manifest.tensors['model.layers.0.self_attn.q_proj.input_max'].shape, []);
assert.equal(bundle.manifest.quantizationInfo.weights, 'f16');
assert.equal(bundle.manifest.quantization, 'F16');
assert.ok(bundle.manifest.groups?.embed);
assert.ok(bundle.manifest.groups?.head);
assert.equal(bundle.manifest.metadata?.sourceRuntime?.mode, 'direct-source');
assert.equal(bundle.manifest.metadata?.sourceRuntime?.schema, 'direct-source/v1');
assert.equal(bundle.manifest.metadata?.sourceRuntime?.schemaVersion, 1);
assert.equal(bundle.manifest.metadata?.sourceRuntime?.pathSemantics, 'runtime-local');

const shardData = new Map([
  ['weights_a.safetensors', new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7])],
  ['weights_b.safetensors', new Uint8Array([8, 9, 10, 11])],
]);

const storageContext = createSourceStorageContext({
  manifest: bundle.manifest,
  shardSources: bundle.shardSources,
  readRange: async (path, offset, length) => {
    const bytes = shardData.get(path);
    if (!bytes) {
      throw new Error(`missing shard ${path}`);
    }
    return bytes.slice(offset, offset + length);
  },
  readText: async (path) => {
    if (path === 'tokenizer.json') {
      return JSON.stringify({ test: true, vocab: {} });
    }
    return null;
  },
  readBinary: async (path) => {
    if (path !== 'tokenizer.model') {
      throw new Error(`missing binary ${path}`);
    }
    return new Uint8Array([1, 2, 3, 4]);
  },
  tokenizerJsonPath: 'tokenizer.json',
  tokenizerModelPath: 'tokenizer.model',
  verifyHashes: false,
});

const range = await storageContext.loadShardRange(0, 2, 3);
assert.deepEqual(Array.from(new Uint8Array(range)), [2, 3, 4]);

const streamed = [];
for await (const chunk of storageContext.streamShardRange(0, 1, 4, { chunkBytes: 2 })) {
  streamed.push(...chunk);
}
assert.deepEqual(streamed, [1, 2, 3, 4]);

assert.deepEqual(await storageContext.loadTokenizerJson(), { test: true, vocab: {} });
const tokenizerModel = await storageContext.loadTokenizerModel();
assert.deepEqual(Array.from(new Uint8Array(tokenizerModel)), [1, 2, 3, 4]);
assert.equal(storageContext.verifyHashes, false);

const missingTokenizerContext = createSourceStorageContext({
  manifest: bundle.manifest,
  shardSources: bundle.shardSources,
  readRange: async () => new Uint8Array([0, 1, 2, 3]),
  readText: async () => null,
  tokenizerJsonPath: 'tokenizer.json',
});
await assert.rejects(
  () => missingTokenizerContext.loadTokenizerJson(),
  /did not return tokenizer JSON data/
);

const verifyContext = createSourceStorageContext({
  manifest: bundle.manifest,
  shardSources: bundle.shardSources,
  readRange: async () => new Uint8Array([1, 2, 3, 4]),
  verifyHashes: true,
});
assert.equal(verifyContext.verifyHashes, true);
assert.equal(verifyContext.loadShardRange, null);
assert.equal(verifyContext.streamShardRange, null);

const malformedRangeContext = createSourceStorageContext({
  manifest: bundle.manifest,
  shardSources: bundle.shardSources,
  readRange: async () => 'invalid-range-payload',
});
await assert.rejects(
  () => malformedRangeContext.loadShardRange(0, 0, 2),
  /must return ArrayBuffer or Uint8Array/
);

const malformedStreamContext = createSourceStorageContext({
  manifest: bundle.manifest,
  shardSources: bundle.shardSources,
  readRange: async () => new Uint8Array([0, 1, 2, 3]),
  streamRange: async function* () {
    yield 'bad-stream-chunk';
  },
});
await assert.rejects(
  async () => {
    for await (const _chunk of malformedStreamContext.streamShardRange(0, 0, 4, { chunkBytes: 2 })) {
      // iterate to trigger loader validation
    }
  },
  /must return ArrayBuffer or Uint8Array/
);

const partialTokenizerContext = createSourceStorageContext({
  manifest: bundle.manifest,
  shardSources: bundle.shardSources,
  readRange: async () => new Uint8Array([0, 1, 2, 3]),
  readBinary: async () => new Uint8Array(0),
  tokenizerModelPath: 'tokenizer.model',
});
await assert.rejects(
  () => partialTokenizerContext.loadTokenizerModel(),
  /empty tokenizer model payload/
);

console.log('source-runtime-bundle.test: ok');
