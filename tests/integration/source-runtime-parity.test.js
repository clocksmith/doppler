import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const { runNodeCommand } = await import('../../src/tooling/node-command-runner.js');
const { bootstrapNodeWebGPU } = await import('../../src/tooling/node-webgpu.js');
const { convertSafetensorsDirectory } = await import('../../src/tooling/node-converter.js');

const PROMPT = 'Describe Doppler in three words.';

const executionSessionDefaults = {
  compute: {
    defaults: {
      activationDtype: 'f16',
      mathDtype: 'f16',
      accumDtype: 'f32',
      outputDtype: 'f16',
    },
    kernelProfiles: [],
  },
  kvcache: null,
  decodeLoop: null,
};

const castOnlyExecution = {
  steps: [
    {
      id: 'cast.layer.identity',
      op: 'cast',
      phase: 'both',
      section: 'layer',
      src: 'state',
      dst: 'state',
      layers: 'all',
      toDtype: 'f16',
    },
  ],
};

function encodeJson(value) {
  return new TextEncoder().encode(JSON.stringify(value));
}

function toModelUrl(value) {
  const resolved = path.resolve(value);
  const asUrl = pathToFileURL(resolved).toString();
  return asUrl.endsWith('/') ? asUrl : `${asUrl}/`;
}

function normalizeOutput(value) {
  return String(value || '').replace(/\s+/g, ' ').trim();
}

function buildGemma3SafetensorsFixtureBytes() {
  const tensorShapes = new Map([
    ['model.embed_tokens.weight', [4, 2]],
    ['model.layers.0.input_layernorm.weight', [2]],
    ['model.layers.0.pre_feedforward_layernorm.weight', [2]],
    ['model.layers.0.post_attention_layernorm.weight', [2]],
    ['model.layers.0.post_feedforward_layernorm.weight', [2]],
    ['model.layers.0.self_attn.q_norm.weight', [2]],
    ['model.layers.0.self_attn.k_norm.weight', [2]],
    ['model.layers.0.self_attn.q_proj.weight', [2, 2]],
    ['model.layers.0.self_attn.k_proj.weight', [2, 2]],
    ['model.layers.0.self_attn.v_proj.weight', [2, 2]],
    ['model.layers.0.self_attn.o_proj.weight', [2, 2]],
    ['model.layers.0.mlp.gate_proj.weight', [2, 2]],
    ['model.layers.0.mlp.up_proj.weight', [2, 2]],
    ['model.layers.0.mlp.down_proj.weight', [2, 2]],
    ['model.norm.weight', [2]],
    ['lm_head.weight', [4, 2]],
  ]);

  const header = {};
  let offset = 0;
  for (const [name, shape] of tensorShapes.entries()) {
    const elements = shape.reduce((product, value) => product * value, 1);
    const bytes = elements * 2;
    header[name] = {
      dtype: 'F16',
      shape,
      data_offsets: [offset, offset + bytes],
    };
    offset += bytes;
  }

  const headerBytes = encodeJson(header);
  const prefix = new ArrayBuffer(8);
  new DataView(prefix).setBigUint64(0, BigInt(headerBytes.byteLength), true);
  const payload = new Uint8Array(offset);
  for (let i = 0; i < payload.byteLength; i += 1) {
    payload[i] = (i * 17) % 251;
  }
  const out = new Uint8Array(8 + headerBytes.byteLength + payload.byteLength);
  out.set(new Uint8Array(prefix), 0);
  out.set(headerBytes, 8);
  out.set(payload, 8 + headerBytes.byteLength);
  return out;
}

function writeGemma3SourceFixture(fixtureDir) {
  writeFileSync(path.join(fixtureDir, 'config.json'), JSON.stringify({
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
  }), 'utf8');
  writeFileSync(path.join(fixtureDir, 'model.safetensors'), buildGemma3SafetensorsFixtureBytes());
  writeFileSync(path.join(fixtureDir, 'tokenizer.json'), JSON.stringify({
    version: '1.0',
    model: {
      vocab: {
        '<pad>': 0,
        '<bos>': 1,
        '<eos>': 2,
        'doppler': 3,
      },
    },
    special_tokens: {
      bos_token: '<bos>',
      eos_token: '<eos>',
      pad_token: '<pad>',
    },
    added_tokens_decoder: {
      '2': { content: '<eos>' },
    },
  }), 'utf8');
}

const fixtureDir = mkdtempSync(path.join(tmpdir(), 'doppler-source-parity-gemma3-'));
const outputDir = path.join(fixtureDir, 'rdrr');

try {
  writeGemma3SourceFixture(fixtureDir);

  let webgpuReady = false;
  try {
    await bootstrapNodeWebGPU();
    webgpuReady = typeof globalThis.navigator !== 'undefined' && !!globalThis.navigator.gpu;
  } catch {
    webgpuReady = false;
  }

  if (!webgpuReady) {
    console.log('source-runtime-parity.test: skipped (no WebGPU runtime)');
  } else {
    await convertSafetensorsDirectory({
      inputDir: fixtureDir,
      converterConfig: {
        output: {
          modelBaseId: 'gemma-3-source-parity-fixture',
          dir: outputDir,
        },
        quantization: {
          weights: 'f16',
        },
        inference: {
          sessionDefaults: executionSessionDefaults,
          execution: castOnlyExecution,
        },
      },
      execution: {
        workers: 1,
      },
    });

    const runtimeConfig = {
      loading: {
        shardCache: {
          verifyHashes: true,
        },
      },
      inference: {
        prompt: PROMPT,
        batching: {
          maxTokens: 8,
        },
        sampling: {
          temperature: 0,
          topP: 1,
          topK: 1,
          repetitionPenalty: 1,
          greedyThreshold: 0,
        },
      },
    };

    const common = {
      command: 'debug',
      modelId: 'gemma-3-source-parity-fixture',
      runtimeConfig,
      captureOutput: true,
    };

    const rdrr = await runNodeCommand({
      ...common,
      modelUrl: toModelUrl(outputDir),
      loadMode: 'http',
    });
    const source = await runNodeCommand({
      ...common,
      modelUrl: fixtureDir,
      loadMode: 'memory',
    });

    const rdrrResult = rdrr?.result ?? null;
    const sourceResult = source?.result ?? null;
    assert.ok(rdrrResult, 'RDRR run result is required.');
    assert.ok(sourceResult, 'Direct-source run result is required.');

    const rdrrOutput = normalizeOutput(rdrrResult.output);
    const sourceOutput = normalizeOutput(sourceResult.output);
    assert.ok(rdrrOutput.length > 0, 'RDRR run did not produce output.');
    assert.ok(sourceOutput.length > 0, 'Direct-source run did not produce output.');

    const rdrrTokens = Number(rdrrResult.metrics?.tokensGenerated || 0);
    const sourceTokens = Number(sourceResult.metrics?.tokensGenerated || 0);
    assert.ok(rdrrTokens > 0, 'RDRR run produced zero tokens.');
    assert.ok(sourceTokens > 0, 'Direct-source run produced zero tokens.');
    assert.equal(sourceTokens, rdrrTokens, 'Direct-source token count must match RDRR exactly.');
    assert.equal(sourceOutput, rdrrOutput, 'Direct-source output must match RDRR exactly.');
  }
} finally {
  rmSync(fixtureDir, { recursive: true, force: true });
}

console.log('source-runtime-parity.test: ok');
