import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { tmpdir } from 'node:os';

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
      dtype: 'F16',
      shape: [2, 2],
      data_offsets: [8, 16],
    },
  };
  const headerBytes = encodeJson(header);
  const prefix = new ArrayBuffer(8);
  new DataView(prefix).setBigUint64(0, BigInt(headerBytes.byteLength), true);
  const data = new Uint8Array(16);
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
try {
  writeFileSync(path.join(fixtureDir, 'config.json'), JSON.stringify({
    architectures: ['Gemma2ForCausalLM'],
    model_type: 'gemma2',
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
  });
  assert.ok(bundle);
  assert.equal(bundle.sourceKind, 'safetensors');
  assert.equal(bundle.manifest.modelId, 'node-source-runtime-test');
  assert.equal(bundle.storageContext.verifyHashes, false);

  const range = await bundle.storageContext.loadShardRange(0, 0, 8);
  assert.equal(new Uint8Array(range).byteLength, 8);

  const tokenizer = await bundle.storageContext.loadTokenizerJson();
  assert.equal(typeof tokenizer, 'object');
} finally {
  rmSync(fixtureDir, { recursive: true, force: true });
}

console.log('node-source-runtime.test: ok');

