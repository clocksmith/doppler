import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { tmpdir } from 'node:os';
import { totalmem } from 'node:os';
import { buildGemma4LiteRTPackedFixture } from '../helpers/gemma4-litert-fixture.js';
import { buildTfliteFixture, FIXTURE_TFLITE_TENSOR_TYPE } from '../helpers/tflite-fixture.js';
import {
  FIXTURE_LITERTLM_SECTION_TYPE,
  buildLiteRTLmFixture,
} from '../helpers/litert-package-fixture.js';

const { resolveNodeSourceRuntimeBundle } = await import('../../src/tooling/node-source-runtime.js');

function encodeJson(value) {
  return new TextEncoder().encode(JSON.stringify(value));
}

function buildSafetensorsBytes() {
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
    'model.layers.0.self_attn.q_proj.input_max': {
      dtype: 'BF16',
      shape: [],
      data_offsets: [16, 18],
    },
  };
  const headerBytes = encodeJson(header);
  const prefix = new ArrayBuffer(8);
  new DataView(prefix).setBigUint64(0, BigInt(headerBytes.byteLength), true);
  const data = new Uint8Array(18);
  for (let i = 0; i < data.byteLength; i++) {
    data[i] = i;
  }
  const out = new Uint8Array(8 + headerBytes.byteLength + data.byteLength);
  out.set(new Uint8Array(prefix), 0);
  out.set(headerBytes, 8);
  out.set(data, 8 + headerBytes.byteLength);
  return out;
}

const fixtureDir = mkdtempSync(path.join(tmpdir(), 'doppler-node-source-runtime-'));
const tfliteFixtureDir = mkdtempSync(path.join(tmpdir(), 'doppler-node-source-runtime-tflite-'));
const litertTaskFixtureDir = mkdtempSync(path.join(tmpdir(), 'doppler-node-source-runtime-task-'));
const litertLmFixtureDir = mkdtempSync(path.join(tmpdir(), 'doppler-node-source-runtime-litertlm-'));
const litertPackedBytes = buildGemma4LiteRTPackedFixture({ profileAligned: true });
try {
  writeFileSync(path.join(fixtureDir, 'config.json'), JSON.stringify({
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
  }), 'utf8');
  writeFileSync(path.join(fixtureDir, 'model.safetensors'), buildSafetensorsBytes());
  writeFileSync(path.join(fixtureDir, 'tokenizer.json'), JSON.stringify({
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
  }), 'utf8');

  const bundle = await resolveNodeSourceRuntimeBundle({
    inputPath: fixtureDir,
    modelId: 'node-source-runtime-test',
    verifyHashes: true,
    runtimeConfig: {
      loading: {
        shardCache: {
          verifyHashes: true,
        },
        storage: {
          backend: {
            streaming: {
              readChunkBytes: 1024,
              maxInFlightBytes: 2048,
            },
          },
        },
        memoryManagement: {
          budget: {
            enabled: true,
            maxResidentBytes: null,
            systemMemoryFraction: 0.5,
            reserveBytes: 2 * 1024 * 1024 * 1024,
            minimumBudgetBytes: 512 * 1024 * 1024,
          },
        },
      },
    },
  });
  assert.ok(bundle, 'node source runtime should synthesize a direct-source bundle');
  assert.equal(bundle.sourceKind, 'safetensors');
  assert.equal(bundle.model, bundle.manifest);
  assert.equal(bundle.model.kind, 'runtime-model');
  assert.equal(bundle.manifest.modelId, 'node-source-runtime-test');
  assert.equal(bundle.manifest.inference?.execution?.kernels?.embed?.kernel, 'gather_f16.wgsl');
  assert.ok(bundle.storageContext, 'node source runtime should create a storage context');
  assert.ok(
    Number.isFinite(bundle.resolvedMemoryBudgetBytes) && bundle.resolvedMemoryBudgetBytes > 0,
    'node source runtime should resolve an absolute resident memory budget'
  );

  const shard = await bundle.storageContext.loadShard(0);
  assert.ok(new Uint8Array(shard).byteLength > 0, 'source-runtime storage context should load shard bytes');

  await assert.rejects(
    () => resolveNodeSourceRuntimeBundle({
      inputPath: fixtureDir,
      modelId: 'node-source-runtime-test-budget-reject',
      verifyHashes: true,
      runtimeConfig: {
        loading: {
          shardCache: {
            verifyHashes: true,
          },
          storage: {
            backend: {
              streaming: {
                readChunkBytes: 1024,
                maxInFlightBytes: 2048,
              },
            },
          },
          memoryManagement: {
            budget: {
              enabled: true,
              maxResidentBytes: Math.max(1, Math.floor(totalmem() * 0.000001)),
              systemMemoryFraction: null,
              reserveBytes: 0,
              minimumBudgetBytes: 1,
            },
          },
        },
      },
    }),
    /direct-source load exceeds resident memory budget/
  );

  writeFileSync(path.join(tfliteFixtureDir, 'config.json'), JSON.stringify({
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
  }), 'utf8');
  writeFileSync(path.join(tfliteFixtureDir, 'tokenizer.json'), JSON.stringify({
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
  }), 'utf8');
  const tfliteBytes = buildTfliteFixture({
    description: 'node-source-runtime-tflite',
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
  writeFileSync(path.join(tfliteFixtureDir, 'model.tflite'), tfliteBytes);
  const tfliteBundle = await resolveNodeSourceRuntimeBundle({
    inputPath: path.join(tfliteFixtureDir, 'model.tflite'),
    modelId: 'node-source-runtime-test-tflite',
  });
  assert.ok(tfliteBundle, 'node source runtime should synthesize a TFLite direct-source bundle');
  assert.equal(tfliteBundle.sourceKind, 'tflite');
  assert.equal(tfliteBundle.model, tfliteBundle.manifest);
  assert.equal(tfliteBundle.manifest.modelId, 'node-source-runtime-test-tflite');
  assert.equal(
    tfliteBundle.manifest.tensors['model.layers.0.self_attn.q_proj.weight']?.sourceTransform?.kind,
    'affine_dequant'
  );
  const tfliteTokenizer = await tfliteBundle.storageContext.loadTokenizerJson();
  assert.equal(typeof tfliteTokenizer, 'object');

  writeFileSync(path.join(litertTaskFixtureDir, 'gemma-4-e2b-it-web.task'), litertPackedBytes);
  writeFileSync(path.join(litertTaskFixtureDir, 'tokenizer.json'), JSON.stringify({
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
  }), 'utf8');
  const litertTaskBundle = await resolveNodeSourceRuntimeBundle({
    inputPath: path.join(litertTaskFixtureDir, 'gemma-4-e2b-it-web.task'),
    modelId: 'node-source-runtime-test-litert-task',
  });
  assert.ok(litertTaskBundle, 'node source runtime should synthesize a LiteRT task direct-source bundle');
  assert.equal(litertTaskBundle.sourceKind, 'litert-task');
  assert.equal(litertTaskBundle.model, litertTaskBundle.manifest);
  assert.equal(litertTaskBundle.manifest.modelId, 'node-source-runtime-test-litert-task');
  assert.equal(litertTaskBundle.manifest.modelType, 'gemma4');
  assert.equal(
    litertTaskBundle.manifest.inference?.execution?.kernels?.attn_decode?.kernel,
    'attention_decode_online_f16kv.wgsl'
  );
  assert.equal(
    litertTaskBundle.manifest.tensors['model.language_model.layers.0.self_attn.q_proj.weight']?.sourceTransform?.kind,
    'litert_axis_dequant'
  );
  assert.ok(
    litertTaskBundle.manifest.tensors['model.language_model.layers.0.self_attn.q_proj.weight']?.sourceTransform?.sumSource,
    'LiteRT task bundle should preserve sum companions for packed axis-dequant tensors'
  );
  assert.deepEqual(
    litertTaskBundle.manifest.tensors['model.language_model.layers.0.self_attn.o_proj.weight']?.sourceTransform?.storageShape,
    [2048, 1536]
  );
  assert.equal(
    litertTaskBundle.manifest.tensors['model.language_model.layers.0.self_attn.o_proj.weight']?.sourceTransform?.quantAxis,
    0
  );
  assert.deepEqual(
    litertTaskBundle.manifest.tensors['model.language_model.layers.0.mlp.down_proj.weight']?.sourceTransform?.storageShape,
    [6144, 1536]
  );
  assert.equal(
    litertTaskBundle.manifest.tensors['model.language_model.layers.0.mlp.down_proj.weight']?.sourceTransform?.quantAxis,
    0
  );
  assert.deepEqual(
    litertTaskBundle.manifest.tensors['model.language_model.layers.0.per_layer_projection.weight']?.sourceTransform?.storageShape,
    [256, 1536]
  );
  assert.equal(
    litertTaskBundle.manifest.tensors['model.language_model.layers.0.per_layer_projection.weight']?.sourceTransform?.quantAxis,
    0
  );
  assert.deepEqual(
    litertTaskBundle.manifest.tensors['model.language_model.layers.0.self_attn.q_proj.weight']?.shape,
    [2048, 1536]
  );
  assert.equal(
    litertTaskBundle.manifest.tensors['model.language_model.layers.0.per_layer_input_gate.weight']?.role,
    'other'
  );
  assert.ok(
    litertTaskBundle.manifest.tensors['model.language_model.layers.34.embed_tokens_per_layer.weight'],
    'LiteRT task bundle should normalize split per-layer embedding tables'
  );
  assert.equal(
    litertTaskBundle.manifest.tensors['model.language_model.layers.34.embed_tokens_per_layer.weight']?.sourceTransform?.rowSumSource ?? null,
    null,
    'LiteRT task per-layer embedding tables should keep the no-row-sum contract'
  );
  const litertTaskTokenizer = await litertTaskBundle.storageContext.loadTokenizerJson();
  assert.equal(typeof litertTaskTokenizer, 'object');

  writeFileSync(path.join(litertLmFixtureDir, 'gemma-4-e2b-it.litertlm'), buildLiteRTLmFixture({
    sections: [
      {
        dataType: FIXTURE_LITERTLM_SECTION_TYPE.TFLiteModel,
        data: litertPackedBytes,
      },
      {
        dataType: FIXTURE_LITERTLM_SECTION_TYPE.SP_Tokenizer,
        data: Uint8Array.from([1, 3, 5, 7]),
      },
    ],
  }));
  const litertLmBundle = await resolveNodeSourceRuntimeBundle({
    inputPath: path.join(litertLmFixtureDir, 'gemma-4-e2b-it.litertlm'),
    modelId: 'node-source-runtime-test-litertlm',
  });
  assert.ok(litertLmBundle, 'node source runtime should synthesize a LiteRT-LM direct-source bundle');
  assert.equal(litertLmBundle.sourceKind, 'litertlm');
  assert.equal(litertLmBundle.model, litertLmBundle.manifest);
  assert.equal(litertLmBundle.manifest.modelType, 'gemma4');
  assert.equal(
    litertLmBundle.manifest.tensors['model.language_model.embed_tokens.weight']?.sourceTransform?.kind,
    'litert_axis_dequant'
  );
  const litertLmTokenizer = await litertLmBundle.storageContext.loadTokenizerModel();
  assert.equal(litertLmTokenizer?.byteLength, 4);

  writeFileSync(path.join(litertLmFixtureDir, 'gemma-4-e2b-it.litertlm'), buildLiteRTLmFixture({
    sections: [
      {
        dataType: FIXTURE_LITERTLM_SECTION_TYPE.TFLiteModel,
        data: tfliteBytes,
      },
      {
        dataType: FIXTURE_LITERTLM_SECTION_TYPE.TFLiteWeights,
        data: Uint8Array.from([9, 9, 9, 9]),
      },
    ],
  }));
  await assert.rejects(
    () => resolveNodeSourceRuntimeBundle({
      inputPath: path.join(litertLmFixtureDir, 'gemma-4-e2b-it.litertlm'),
      modelId: 'node-source-runtime-test-litertlm-external',
    }),
    /External-weight LiteRT-LM packages are not supported yet/
  );
} finally {
  rmSync(fixtureDir, { recursive: true, force: true });
  rmSync(tfliteFixtureDir, { recursive: true, force: true });
  rmSync(litertTaskFixtureDir, { recursive: true, force: true });
  rmSync(litertLmFixtureDir, { recursive: true, force: true });
}

console.log('node-source-runtime.test: ok');
