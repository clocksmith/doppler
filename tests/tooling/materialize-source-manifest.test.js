import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { tmpdir } from 'node:os';

import { DEFAULT_MANIFEST_INFERENCE } from '../../src/config/schema/index.js';
import { buildSourceRuntimeBundle } from '../../src/tooling/source-runtime-bundle.js';
import { materializeSourceRuntimeManifest } from '../../src/tooling/source-runtime-materializer.js';
import { assertManifestArtifactIntegrity } from '../helpers/local-model-fixture.js';

function encodeJson(value) {
  return new TextEncoder().encode(JSON.stringify(value));
}

function computeSha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function buildSafetensorsFixture() {
  const tensorNames = [
    'model.embed_tokens.weight',
    'model.layers.0.input_layernorm.weight',
    'model.layers.0.pre_feedforward_layernorm.weight',
    'model.layers.0.post_attention_layernorm.weight',
    'model.layers.0.post_feedforward_layernorm.weight',
    'model.layers.0.self_attn.q_norm.weight',
    'model.layers.0.self_attn.k_norm.weight',
    'model.layers.0.self_attn.q_proj.weight',
    'model.layers.0.self_attn.k_proj.weight',
    'model.layers.0.self_attn.v_proj.weight',
    'model.layers.0.self_attn.o_proj.weight',
    'model.layers.0.mlp.gate_proj.weight',
    'model.layers.0.mlp.up_proj.weight',
    'model.layers.0.mlp.down_proj.weight',
    'model.norm.weight',
    'lm_head.weight',
  ];
  const header = {};
  const tensors = [];
  let offset = 0;
  for (const name of tensorNames) {
    const isMatrix = name.includes('embed_tokens') || name === 'lm_head.weight';
    const shape = isMatrix ? [4, 2] : [2];
    const size = isMatrix ? 16 : 4;
    header[name] = {
      dtype: 'F16',
      shape,
      data_offsets: [offset, offset + size],
    };
    tensors.push({
      name,
      dtype: 'F16',
      shape,
      size,
      offset,
    });
    offset += size;
  }
  const headerBytes = encodeJson(header);
  const prefix = new ArrayBuffer(8);
  new DataView(prefix).setBigUint64(0, BigInt(headerBytes.byteLength), true);
  const payload = new Uint8Array(offset);
  for (let i = 0; i < payload.byteLength; i += 1) {
    payload[i] = i % 251;
  }
  const out = new Uint8Array(8 + headerBytes.byteLength + payload.byteLength);
  out.set(new Uint8Array(prefix), 0);
  out.set(headerBytes, 8);
  out.set(payload, 8 + headerBytes.byteLength);
  return {
    bytes: out,
    tensors,
  };
}

const fixtureDir = mkdtempSync(path.join(tmpdir(), 'doppler-direct-source-artifact-'));
try {
  const config = {
    architectures: ['Gemma3ForCausalLM'],
    model_type: 'gemma3_text',
    num_hidden_layers: 1,
    hidden_size: 2,
    num_attention_heads: 1,
    num_key_value_heads: 1,
    head_dim: 2,
    intermediate_size: 2,
    vocab_size: 4,
    max_position_embeddings: 8,
    bos_token_id: 1,
    eos_token_id: 2,
    rms_norm_eps: 1e-6,
  };
  const tokenizerJson = {
    version: '1.0',
    model: {
      vocab: {
        '<pad>': 0,
        '<bos>': 1,
        '<eos>': 2,
        'hello': 3,
      },
    },
    special_tokens: {
      bos_token: '<bos>',
      eos_token: '<eos>',
    },
    added_tokens_decoder: {
      '2': { content: '<eos>' },
    },
  };
  const modelPath = path.join(fixtureDir, 'model.safetensors');
  const tokenizerPath = path.join(fixtureDir, 'tokenizer.json');
  const { bytes, tensors } = buildSafetensorsFixture();

  writeFileSync(path.join(fixtureDir, 'config.json'), JSON.stringify({
    ...config,
  }), 'utf8');
  writeFileSync(modelPath, bytes);
  writeFileSync(tokenizerPath, JSON.stringify(tokenizerJson), 'utf8');

  const manifestPath = path.join(fixtureDir, 'manifest.json');
  const bundle = await buildSourceRuntimeBundle({
    modelId: 'gemma-3-direct-source-artifact-test',
    modelType: 'transformer',
    architecture: {
      numLayers: 1,
      hiddenSize: 2,
      intermediateSize: 2,
      numAttentionHeads: 1,
      numKeyValueHeads: 1,
      headDim: 2,
      vocabSize: 4,
      maxSeqLen: 8,
      ropeTheta: 10000,
    },
    architectureHint: 'gemma3',
    rawConfig: config,
    inference: JSON.parse(JSON.stringify(DEFAULT_MANIFEST_INFERENCE)),
    tensors: tensors.map((tensor) => ({
      ...tensor,
      sourcePath: modelPath,
    })),
    sourceFiles: [
      {
        path: modelPath,
        size: bytes.byteLength,
        hash: computeSha256(bytes),
        hashAlgorithm: 'sha256',
      },
    ],
    sourceQuantization: 'f16',
    tokenizerJson,
    tokenizerJsonPath: tokenizerPath,
  });
  assert.ok(bundle);
  writeFileSync(
    manifestPath,
    `${JSON.stringify(materializeSourceRuntimeManifest(bundle.manifest, fixtureDir), null, 2)}\n`,
    'utf8'
  );
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  assert.equal(manifest.modelId, 'gemma-3-direct-source-artifact-test');
  assert.equal(manifest.metadata?.sourceRuntime?.pathSemantics, 'artifact-relative');
  assert.equal(manifest.metadata?.sourceRuntime?.tokenizer?.jsonPath, 'tokenizer.json');
  assert.equal(manifest.metadata?.sourceRuntime?.sourceFiles?.[0]?.path, 'model.safetensors');
  await assertManifestArtifactIntegrity(manifestPath);

  const relativeManifest = materializeSourceRuntimeManifest(
    {
      ...bundle.manifest,
      metadata: {
        ...bundle.manifest.metadata,
        sourceRuntime: {
          ...bundle.manifest.metadata.sourceRuntime,
          sourceFiles: [
            {
              ...bundle.manifest.metadata.sourceRuntime.sourceFiles[0],
              path: 'model.safetensors',
            },
          ],
          auxiliaryFiles: [
            {
              path: 'tokenizer.json',
              size: encodeJson(tokenizerJson).byteLength,
              hash: computeSha256(encodeJson(tokenizerJson)),
              hashAlgorithm: 'sha256',
              kind: 'tokenizer_json',
            },
          ],
          tokenizer: {
            jsonPath: 'tokenizer.json',
            configPath: null,
            modelPath: null,
          },
        },
      },
    },
    fixtureDir
  );
  assert.equal(relativeManifest.metadata?.sourceRuntime?.sourceFiles?.[0]?.path, 'model.safetensors');
  assert.equal(relativeManifest.metadata?.sourceRuntime?.auxiliaryFiles?.[0]?.path, 'tokenizer.json');
  assert.equal(relativeManifest.metadata?.sourceRuntime?.tokenizer?.jsonPath, 'tokenizer.json');
} finally {
  rmSync(fixtureDir, { recursive: true, force: true });
}

console.log('materialize-source-manifest.test: ok');
