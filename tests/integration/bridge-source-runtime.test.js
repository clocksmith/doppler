import assert from 'node:assert/strict';
import { buildGemma4LiteRTPackedFixture } from '../helpers/gemma4-litert-fixture.js';
import { buildTfliteFixture, FIXTURE_TFLITE_TENSOR_TYPE } from '../helpers/tflite-fixture.js';
import {
  FIXTURE_LITERTLM_SECTION_TYPE,
  buildLiteRTLmFixture,
} from '../helpers/litert-package-fixture.js';

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
const tfliteBytes = buildTfliteFixture({
  description: 'bridge-source-runtime-tflite',
  tensors: [
    {
      name: 'model.embed_tokens.weight',
      shape: [2, 2],
      type: FIXTURE_TFLITE_TENSOR_TYPE.FLOAT16,
      data: Uint8Array.from([0, 1, 2, 3, 4, 5, 6, 7]),
    },
    {
      name: 'model.layers.0.self_attn.q_proj.weight',
      shape: [2, 2],
      type: FIXTURE_TFLITE_TENSOR_TYPE.INT8,
      data: Uint8Array.from([8, 9, 10, 11]),
      quantization: {
        scales: [0.25],
        zeroPoints: [8],
        quantizedDimension: 0,
      },
    },
  ],
});
const litertPackedBytes = buildGemma4LiteRTPackedFixture({ profileAligned: true });
const litertLmBytes = buildLiteRTLmFixture({
  sections: [
    {
      dataType: FIXTURE_LITERTLM_SECTION_TYPE.TFLiteModel,
      data: litertPackedBytes,
    },
    {
      dataType: FIXTURE_LITERTLM_SECTION_TYPE.SP_Tokenizer,
      data: Uint8Array.from([6, 4, 2, 0]),
    },
  ],
});

const rootPath = '/model-root';
const files = new Map([
  [`${rootPath}/config.json`, configBytes],
  [`${rootPath}/model.safetensors`, safetensorsBytes],
  [`${rootPath}/tokenizer.json`, tokenizerBytes],
  ['/tflite-root/config.json', encodeJson({
    architectures: ['Gemma4ForCausalLM'],
    model_type: 'gemma4_text',
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
  })],
  ['/tflite-root/tokenizer.json', tokenizerBytes],
  ['/tflite-root/model.tflite', tfliteBytes],
  ['/task-root/gemma-4-e2b-it-web.task', litertPackedBytes],
  ['/task-root/tokenizer.json', tokenizerBytes],
  ['/litertlm-root/gemma-4-e2b-it.litertlm', litertLmBytes],
]);

const bridgeClient = {
  async list(path) {
    if (path !== rootPath) {
      if (path === '/tflite-root') {
        return [
          { name: 'config.json', isDir: false, size: files.get('/tflite-root/config.json').byteLength },
          { name: 'tokenizer.json', isDir: false, size: tokenizerBytes.byteLength },
          { name: 'model.tflite', isDir: false, size: tfliteBytes.byteLength },
        ];
      }
      if (path === '/task-root') {
        return [
          { name: 'gemma-4-e2b-it-web.task', isDir: false, size: litertPackedBytes.byteLength },
          { name: 'tokenizer.json', isDir: false, size: tokenizerBytes.byteLength },
        ];
      }
      if (path === '/litertlm-root') {
        return [
          { name: 'gemma-4-e2b-it.litertlm', isDir: false, size: litertLmBytes.byteLength },
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
assert.equal(bundle.model, bundle.manifest);
assert.equal(bundle.model.kind, 'runtime-model');
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

const tfliteBundle = await resolveBridgeSourceRuntimeBundle({
  bridgeClient,
  localPath: '/tflite-root',
  modelId: 'bridge-source-tflite',
  onProgress() {},
});
assert.ok(tfliteBundle);
assert.equal(tfliteBundle.sourceKind, 'tflite');
assert.equal(tfliteBundle.model, tfliteBundle.manifest);
assert.equal(tfliteBundle.manifest.modelId, 'bridge-source-tflite');
assert.equal(
  tfliteBundle.manifest.tensors['model.layers.0.self_attn.q_proj.weight']?.sourceTransform?.kind,
  'affine_dequant'
);
const tfliteTokenizer = await tfliteBundle.storageContext.loadTokenizerJson();
assert.equal(typeof tfliteTokenizer, 'object');

const taskBundle = await resolveBridgeSourceRuntimeBundle({
  bridgeClient,
  localPath: '/task-root/gemma-4-e2b-it-web.task',
  modelId: 'bridge-source-litert-task',
  onProgress() {},
});
assert.ok(taskBundle);
assert.equal(taskBundle.sourceKind, 'litert-task');
assert.equal(taskBundle.model, taskBundle.manifest);
assert.equal(taskBundle.manifest.modelType, 'gemma4');
assert.equal(
  taskBundle.manifest.tensors['model.language_model.layers.0.self_attn.q_proj.weight']?.sourceTransform?.kind,
  'litert_axis_dequant'
);
assert.equal(
  taskBundle.manifest.tensors['model.language_model.layers.0.self_attn.q_proj.weight']?.sourceTransform?.scaleSemantics,
  'step'
);
assert.ok(
  taskBundle.manifest.tensors['model.language_model.layers.0.self_attn.q_proj.weight']?.sourceTransform?.sumSource,
  'bridge LiteRT task bundle should preserve sum companions for packed axis-dequant tensors'
);
assert.equal(
  taskBundle.manifest.tensors['model.language_model.per_layer_model_projection.weight']?.sourceTransform?.scaleSemantics,
  'step'
);
assert.equal(
  taskBundle.manifest.tensors['model.language_model.per_layer_model_projection.input_activation_static_scale']?.dtype,
  'F32'
);
assert.equal(
  taskBundle.manifest.tensors['model.language_model.per_layer_model_projection.output_activation_static_scale']?.dtype,
  'F32'
);
assert.deepEqual(
  taskBundle.manifest.tensors['model.language_model.layers.0.self_attn.o_proj.weight']?.sourceTransform?.storageShape,
  [2048, 1536]
);
assert.equal(
  taskBundle.manifest.tensors['model.language_model.layers.0.self_attn.o_proj.weight']?.sourceTransform?.quantAxis,
  0
);
assert.deepEqual(
  taskBundle.manifest.tensors['model.language_model.layers.0.mlp.down_proj.weight']?.sourceTransform?.storageShape,
  [6144, 1536]
);
assert.equal(
  taskBundle.manifest.tensors['model.language_model.layers.0.mlp.down_proj.weight']?.sourceTransform?.quantAxis,
  0
);
assert.deepEqual(
  taskBundle.manifest.tensors['model.language_model.layers.0.per_layer_projection.weight']?.sourceTransform?.storageShape,
  [256, 1536]
);
assert.equal(
  taskBundle.manifest.tensors['model.language_model.layers.0.per_layer_projection.weight']?.sourceTransform?.quantAxis,
  0
);
assert.equal(
  taskBundle.manifest.tensors['model.language_model.layers.0.per_layer_input_gate.weight']?.role,
  'matmul'
);
assert.ok(taskBundle.manifest.tensors['model.language_model.layers.34.embed_tokens_per_layer.weight']);
assert.equal(
  taskBundle.manifest.tensors['model.language_model.layers.34.embed_tokens_per_layer.weight']?.sourceTransform?.rowSumSource ?? null,
  null
);
assert.equal(
  taskBundle.manifest.tensors['model.language_model.layers.34.embed_tokens_per_layer.weight']?.sourceTransform?.storageEncoding,
  'signed'
);
assert.deepEqual(
  taskBundle.manifest.tensors['model.language_model.layers.34.embed_tokens_per_layer.weight']?.sourceTransform?.storageShape,
  [262144, 256]
);
assert.equal(
  taskBundle.manifest.tensors['model.language_model.layers.34.embed_tokens_per_layer.weight']?.sourceTransform?.scaleSemantics,
  'step'
);
assert.equal(
  taskBundle.manifest.tensors['model.language_model.layers.34.embed_tokens_per_layer.weight']?.sourceTransform?.quantAxis,
  1
);
assert.equal(
  taskBundle.manifest.tensors['model.language_model.layers.34.embed_tokens_per_layer.weight']?.sourceTransform?.scaleDivisor ?? null,
  null
);
const taskTokenizer = await taskBundle.storageContext.loadTokenizerJson();
assert.equal(typeof taskTokenizer, 'object');

const litertLmBundle = await resolveBridgeSourceRuntimeBundle({
  bridgeClient,
  localPath: '/litertlm-root/gemma-4-e2b-it.litertlm',
  modelId: 'bridge-source-litertlm',
  onProgress() {},
});
assert.ok(litertLmBundle);
assert.equal(litertLmBundle.sourceKind, 'litertlm');
assert.equal(litertLmBundle.model, litertLmBundle.manifest);
assert.equal(litertLmBundle.manifest.modelType, 'gemma4');
assert.equal(
  litertLmBundle.manifest.tensors['model.language_model.embed_tokens.weight']?.sourceTransform?.kind,
  'litert_axis_dequant'
);
assert.equal(
  litertLmBundle.manifest.tensors['model.language_model.embed_tokens.weight']?.sourceTransform?.scaleSemantics,
  'step'
);
const litertLmTokenizer = await litertLmBundle.storageContext.loadTokenizerModel();
assert.equal(litertLmTokenizer?.byteLength, 4);

console.log('bridge-source-runtime.test: ok');
