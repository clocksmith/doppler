import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

const { bootstrapNodeWebGPU } = await import('../../src/tooling/node-webgpu.js');
const { convertSafetensorsDirectory } = await import('../../src/tooling/node-converter.js');
const { resolveNodeSourceRuntimeBundle } = await import('../../src/tooling/node-source-runtime.js');
const { parseManifest } = await import('../../src/formats/rdrr/index.js');
const { createPipeline } = await import('../../src/inference/pipelines/text.js');
const { restorePipelineContexts } = await import('../../src/inference/pipelines/context.js');
const { initDevice } = await import('../../src/gpu/device.js');

const PROMPT = 'doppler';
const MODEL_ID = 'llama-3-source-parity-fixture';
const LOGITS_TOLERANCE = 1e-3;

const executionSessionDefaults = {
  compute: {
    defaults: {
      activationDtype: 'f32',
      mathDtype: 'f32',
      accumDtype: 'f32',
      outputDtype: 'f32',
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
      toDtype: 'f32',
    },
  ],
};

function encodeJson(value) {
  return new TextEncoder().encode(JSON.stringify(value));
}

function toArrayBuffer(value) {
  if (value instanceof ArrayBuffer) {
    return value;
  }
  if (ArrayBuffer.isView(value)) {
    return value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength);
  }
  throw new Error('Expected ArrayBuffer or ArrayBufferView.');
}

function coerceLogitsVector(value, label) {
  if (value instanceof Float32Array) {
    assert.ok(value.length > 0, `${label} must not be empty.`);
    return value;
  }
  if (ArrayBuffer.isView(value)) {
    assert.ok(value.length > 0, `${label} must not be empty.`);
    return Float32Array.from(value);
  }
  if (Array.isArray(value)) {
    assert.ok(value.length > 0, `${label} must not be empty.`);
    return Float32Array.from(value);
  }
  throw new Error(`${label} must be a numeric logits vector.`);
}

function assertLogitsClose(actual, expected, label) {
  assert.equal(actual.length, expected.length, `${label} length must match.`);
  for (let idx = 0; idx < actual.length; idx += 1) {
    const delta = Math.abs(actual[idx] - expected[idx]);
    assert.ok(
      delta <= LOGITS_TOLERANCE,
      `${label} mismatch at index ${idx}: expected ${expected[idx]}, got ${actual[idx]} (delta=${delta}).`
    );
  }
}

function disposePrefillSnapshot(result) {
  const cache = result?.cache;
  if (cache && typeof cache.clear === 'function') {
    cache.clear();
  }
}

function createLocalRdrrStorageContext(modelDir, manifest) {
  const tokenizerPath = manifest?.tokenizer?.file
    ? path.join(modelDir, manifest.tokenizer.file)
    : null;

  return {
    verifyHashes: true,
    async loadShard(index) {
      const shard = manifest?.shards?.[index];
      if (!shard?.filename) {
        throw new Error(`Manifest shard ${index} is missing filename.`);
      }
      return toArrayBuffer(readFileSync(path.join(modelDir, shard.filename)));
    },
    async loadTokenizerJson() {
      if (!tokenizerPath) {
        throw new Error('Manifest tokenizer.file is required for RDRR parity test.');
      }
      return JSON.parse(readFileSync(tokenizerPath, 'utf8'));
    },
  };
}

function cleanupPipeline(pipeline, prefill) {
  disposePrefillSnapshot(prefill);
  pipeline?.reset?.();
  pipeline?.releaseGPUResources?.();
  restorePipelineContexts(pipeline);
}

function snapshotPrefillResult(result, label) {
  assert.ok(result, `${label} prefill result is required.`);
  return {
    tokens: Array.isArray(result.tokens) ? [...result.tokens] : Array.from(result.tokens ?? []),
    seqLen: result.seqLen,
    logits: Float32Array.from(coerceLogitsVector(result.logits, `${label} logits`)),
  };
}

function buildLlama3SafetensorsFixtureBytes() {
  const tensorShapes = new Map([
    ['model.embed_tokens.weight', [5, 2]],
    ['model.layers.0.input_layernorm.weight', [2]],
    ['model.layers.0.self_attn.q_proj.weight', [2, 2]],
    ['model.layers.0.self_attn.k_proj.weight', [2, 2]],
    ['model.layers.0.self_attn.v_proj.weight', [2, 2]],
    ['model.layers.0.self_attn.o_proj.weight', [2, 2]],
    ['model.layers.0.mlp.gate_proj.weight', [2, 2]],
    ['model.layers.0.mlp.up_proj.weight', [2, 2]],
    ['model.layers.0.mlp.down_proj.weight', [2, 2]],
    ['model.norm.weight', [2]],
    ['lm_head.weight', [5, 2]],
  ]);

  const header = {};
  let offset = 0;
  for (const [name, shape] of tensorShapes.entries()) {
    const elements = shape.reduce((product, value) => product * value, 1);
    const bytes = elements * 4;
    header[name] = {
      dtype: 'F32',
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

function writeLlama3SourceFixture(fixtureDir) {
  writeFileSync(path.join(fixtureDir, 'config.json'), JSON.stringify({
    architectures: ['LlamaForCausalLM'],
    model_type: 'llama',
    num_hidden_layers: 1,
    hidden_size: 2,
    num_attention_heads: 1,
    num_key_value_heads: 1,
    head_dim: 2,
    intermediate_size: 2,
    vocab_size: 5,
    max_position_embeddings: 8,
    bos_token_id: 0,
    eos_token_id: 1,
    rms_norm_eps: 1e-5,
    rope_theta: 500000,
  }), 'utf8');
  writeFileSync(path.join(fixtureDir, 'model.safetensors'), buildLlama3SafetensorsFixtureBytes());
  writeFileSync(path.join(fixtureDir, 'tokenizer.json'), JSON.stringify({
    version: '1.0',
    model: {
      type: 'BPE',
      vocab: {
        '<|begin_of_text|>': 0,
        '<|end_of_text|>': 1,
        '<|eot_id|>': 2,
        'doppler': 3,
        '<unk>': 4,
      },
    },
    special_tokens: {
      bos_token: '<|begin_of_text|>',
      eos_token: '<|end_of_text|>',
      unk_token: '<unk>',
    },
    added_tokens_decoder: {
      '1': { content: '<|end_of_text|>' },
      '2': { content: '<|eot_id|>' },
    },
  }), 'utf8');
}

const fixtureDir = mkdtempSync(path.join(tmpdir(), 'doppler-source-parity-llama3-'));
const outputDir = path.join(fixtureDir, 'rdrr');

try {
  writeLlama3SourceFixture(fixtureDir);
  await convertSafetensorsDirectory({
    inputDir: fixtureDir,
    converterConfig: {
      output: {
        modelBaseId: MODEL_ID,
        dir: outputDir,
      },
      quantization: {
        weights: 'f32',
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
    shared: {
      debug: {
        pipeline: {
          enabled: true,
        },
      },
    },
    loading: {
      shardCache: {
        verifyHashes: true,
      },
    },
    inference: {
      prompt: PROMPT,
      chatTemplate: {
        enabled: false,
      },
      compute: {
        activationDtype: 'f32',
        keepF32Weights: true,
        rangeAwareSelectiveWidening: {
          enabled: false,
          includeNonFinite: true,
          absThreshold: 65500,
        },
      },
      session: {
        compute: {
          defaults: {
            activationDtype: 'f32',
            mathDtype: 'f32',
            accumDtype: 'f32',
            outputDtype: 'f32',
          },
        },
      },
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

  const sourceBundle = await resolveNodeSourceRuntimeBundle({
    inputPath: fixtureDir,
    modelId: MODEL_ID,
    verifyHashes: true,
  });
  assert.ok(sourceBundle, 'Direct-source runtime bundle is required.');

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
    const device = await initDevice();
    const rdrrManifest = parseManifest(readFileSync(path.join(outputDir, 'manifest.json'), 'utf8'));
    const sourceManifestForParity = {
      ...sourceBundle.manifest,
      inference: rdrrManifest.inference,
    };
    let rdrrSnapshot = null;
    let sourceSnapshot = null;

    let rdrrPipeline = null;
    let rdrrResult = null;
    try {
      rdrrPipeline = await createPipeline(rdrrManifest, {
        runtimeConfig,
        gpu: { device },
        storage: createLocalRdrrStorageContext(outputDir, rdrrManifest),
      });
      rdrrResult = await rdrrPipeline.prefillWithLogits(PROMPT, {
        useChatTemplate: false,
      });
      rdrrSnapshot = snapshotPrefillResult(rdrrResult, 'RDRR');
    } finally {
      cleanupPipeline(rdrrPipeline, rdrrResult);
    }

    let sourcePipeline = null;
    let sourceResult = null;
    try {
      sourcePipeline = await createPipeline(sourceManifestForParity, {
        runtimeConfig,
        gpu: { device },
        storage: sourceBundle.storageContext,
      });
      sourceResult = await sourcePipeline.prefillWithLogits(PROMPT, {
        useChatTemplate: false,
      });
      sourceSnapshot = snapshotPrefillResult(sourceResult, 'Direct-source');
    } finally {
      cleanupPipeline(sourcePipeline, sourceResult);
    }

    assert.ok(rdrrSnapshot, 'RDRR parity snapshot is required.');
    assert.ok(sourceSnapshot, 'Direct-source parity snapshot is required.');
    assert.deepEqual(sourceSnapshot.tokens, rdrrSnapshot.tokens, 'Direct-source prompt token IDs must match RDRR exactly.');
    assert.equal(sourceSnapshot.seqLen, rdrrSnapshot.seqLen, 'Direct-source prefill seqLen must match RDRR exactly.');
    assertLogitsClose(sourceSnapshot.logits, rdrrSnapshot.logits, 'Direct-source logits');
  }
} finally {
  rmSync(fixtureDir, { recursive: true, force: true });
}

console.log('source-runtime-parity.test: ok');
