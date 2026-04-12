import assert from 'node:assert/strict';

const { resolveBridgeSourceRuntimeBundle } = await import('../../src/client/runtime/source-runtime.js');

function encodeJson(value) {
  return new TextEncoder().encode(JSON.stringify(value));
}

function buildSafetensorsFixture() {
  const header = {
    'model.embed_tokens.weight': {
      dtype: 'F16',
      shape: [2, 2],
      data_offsets: [0, 8],
    },
    'model.layers.0.self_attn.q_proj.weight': {
      dtype: 'BF16',
      shape: [2, 2],
      data_offsets: [8, 16],
    },
  };
  const headerBytes = encodeJson(header);
  const prefix = new ArrayBuffer(8);
  new DataView(prefix).setBigUint64(0, BigInt(headerBytes.byteLength), true);
  const dataBytes = new Uint8Array(16);
  for (let i = 0; i < dataBytes.byteLength; i++) {
    dataBytes[i] = i;
  }
  const out = new Uint8Array(8 + headerBytes.byteLength + dataBytes.byteLength);
  out.set(new Uint8Array(prefix), 0);
  out.set(headerBytes, 8);
  out.set(dataBytes, 8 + headerBytes.byteLength);
  return out;
}

const configBytes = encodeJson({
  architectures: ['Gemma3ForCausalLM'],
  model_type: 'gemma3_text',
  num_hidden_layers: 1,
  hidden_size: 2,
  num_attention_heads: 1,
  num_key_value_heads: 1,
  head_dim: 2,
  intermediate_size: 4,
  vocab_size: 2,
  max_position_embeddings: 8,
  rms_norm_eps: 1e-6,
  eos_token_id: 2,
});
const tokenizerBytes = encodeJson({
  model: {
    vocab: {
      '<bos>': 0,
      '<eos>': 1,
    },
  },
  special_tokens: {
    eos_token: '<eos>',
  },
  added_tokens_decoder: {
    '1': { content: '<eos>' },
  },
});
const safetensorsBytes = buildSafetensorsFixture();

const rootPath = '/model-root';
const files = new Map([
  [`${rootPath}/config.json`, configBytes],
  [`${rootPath}/model.safetensors`, safetensorsBytes],
  [`${rootPath}/tokenizer.json`, tokenizerBytes],
  ['/tflite-root/model.tflite', new Uint8Array([1, 2, 3, 4])],
]);

const bridgeClient = {
  async list(path) {
    if (path !== rootPath) {
      if (path === '/tflite-root') {
        return [
          { name: 'model.tflite', isDir: false, size: 4 },
        ];
      }
      throw new Error(`Unexpected list path: ${path}`);
    }
    return [
      { name: 'config.json', isDir: false, size: configBytes.byteLength },
      { name: 'model.safetensors', isDir: false, size: safetensorsBytes.byteLength },
      { name: 'tokenizer.json', isDir: false, size: tokenizerBytes.byteLength },
    ];
  },
  async read(path, offset, length) {
    const bytes = files.get(path);
    if (!bytes) {
      throw new Error(`Missing fixture file: ${path}`);
    }
    const start = Math.max(0, Math.floor(offset));
    const end = Math.min(bytes.byteLength, start + Math.max(0, Math.floor(length)));
    return bytes.slice(start, end);
  },
};

const bundle = await resolveBridgeSourceRuntimeBundle({
  bridgeClient,
  localPath: rootPath,
  modelId: 'bridge-source-test',
  verifyHashes: true,
  onProgress() {},
});

assert.ok(bundle);
assert.equal(bundle.sourceKind, 'safetensors');
assert.equal(bundle.manifest.modelId, 'bridge-source-test');
assert.ok(bundle.storageContext);
assert.equal(bundle.storageContext.verifyHashes, true);

const shard = await bundle.storageContext.loadShard(0);
assert.ok(new Uint8Array(shard).byteLength > 0);
assert.equal(typeof bundle.storageContext.loadShardRange, 'function');

const shardSlice = await bundle.storageContext.loadShardRange(0, 2, 4);
assert.equal(new Uint8Array(shardSlice).byteLength, 4);

const tokenizer = await bundle.storageContext.loadTokenizerJson();
assert.equal(typeof tokenizer, 'object');

await assert.rejects(
  () => resolveBridgeSourceRuntimeBundle({
    bridgeClient,
    localPath: '/tflite-root',
    onProgress() {},
  }),
  /.tflite direct-source artifacts are not implemented yet/
);

console.log('bridge-source-runtime.test: ok');
