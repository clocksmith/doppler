import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

const { probeNodeGPU } = await import('../helpers/gpu-probe.js');
const { DEFAULT_MANIFEST_INFERENCE } = await import('../../src/config/schema/index.js');
const { convertSafetensorsDirectory } = await import('../../src/tooling/node-converter.js');
const { resolveNodeSourceRuntimeBundle } = await import('../../src/tooling/node-source-runtime.js');
const { parseManifest } = await import('../../src/formats/rdrr/index.js');
const { createPipeline } = await import('../../src/inference/pipelines/text.js');
const { restorePipelineContexts } = await import('../../src/inference/pipelines/context.js');
const { initDevice } = await import('../../src/gpu/device.js');
const { createExecutionContractSession } = await import('../helpers/execution-v1-fixtures.js');

const PROMPT = 'doppler';
const MODEL_ID = 'llama-3-source-parity-fixture';
const LOGITS_TOLERANCE = 1e-3;

const executionSessionDefaults = createExecutionContractSession({
  compute: {
    defaults: {
      activationDtype: 'f32',
      mathDtype: 'f32',
      accumDtype: 'f32',
      outputDtype: 'f32',
    },
    kernelProfiles: [],
  },
  kvcache: {
    kvDtype: 'f32',
    layout: 'contiguous',
    pageSize: 256,
    tiering: {
      mode: 'off',
    },
    quantization: {
      mode: 'none',
    },
  },
});

const parityInference = {
  ...DEFAULT_MANIFEST_INFERENCE,
  output: {
    ...DEFAULT_MANIFEST_INFERENCE.output,
    tieWordEmbeddings: false,
    scaleEmbeddings: false,
  },
  chatTemplate: {
    type: 'llama3',
    enabled: false,
  },
};

const parityExecution = {
  kernels: {
    embed: {
      kernel: 'gather.wgsl',
      entry: 'main',
      digest: 'sha256:4b12653c53247b32ebde7f6cf6a989d6248977e3816c761540b990b5f9818cb6',
    },
    rmsnorm: {
      kernel: 'rmsnorm.wgsl',
      entry: 'main',
      digest: 'sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077',
    },
    lm_head: {
      kernel: 'matmul_f32.wgsl',
      entry: 'main',
      digest: 'sha256:b5bb8e3d8014136e33de7935dd2a1f074c988044fe05cf5b559718c6f061eaa8',
    },
  },
  preLayer: [
    ['embed', 'embed', 'embed_tokens'],
  ],
  decode: [],
  prefill: [],
  postLayer: [
    ['final_norm', 'rmsnorm'],
    ['lm_head', 'lm_head', 'lm_head'],
    ['lm_head_prefill', 'lm_head', 'lm_head'],
  ],
  policies: {
    unsupportedPrecision: 'error',
    dtypeTransition: 'require_cast_step',
    unresolvedKernel: 'error',
  },
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
  const tensorValues = new Map([
    ['model.embed_tokens.weight', {
      shape: [5, 2],
      values: [
        -0.50, 0.25,
        0.75, -0.125,
        0.50, 0.375,
        -0.25, 0.625,
        0.125, -0.75,
      ],
    }],
    ['model.layers.0.input_layernorm.weight', {
      shape: [2],
      values: [1.0, 0.875],
    }],
    ['model.layers.0.self_attn.q_proj.weight', {
      shape: [2, 2],
      values: [0.125, -0.25, 0.375, -0.50],
    }],
    ['model.layers.0.self_attn.k_proj.weight', {
      shape: [2, 2],
      values: [0.50, -0.125, -0.375, 0.25],
    }],
    ['model.layers.0.self_attn.v_proj.weight', {
      shape: [2, 2],
      values: [-0.25, 0.50, 0.125, -0.375],
    }],
    ['model.layers.0.self_attn.o_proj.weight', {
      shape: [2, 2],
      values: [0.375, 0.125, -0.50, 0.25],
    }],
    ['model.layers.0.mlp.gate_proj.weight', {
      shape: [2, 2],
      values: [0.25, -0.375, 0.50, -0.125],
    }],
    ['model.layers.0.mlp.up_proj.weight', {
      shape: [2, 2],
      values: [-0.125, 0.50, -0.25, 0.375],
    }],
    ['model.layers.0.mlp.down_proj.weight', {
      shape: [2, 2],
      values: [0.50, 0.25, -0.125, -0.375],
    }],
    ['model.norm.weight', {
      shape: [2],
      values: [1.0, 1.125],
    }],
    ['lm_head.weight', {
      shape: [5, 2],
      values: [
        0.25, -0.50,
        -0.75, 0.375,
        0.50, 0.125,
        -0.25, 0.75,
        0.625, -0.125,
      ],
    }],
  ]);

  const header = {};
  let offset = 0;
  for (const [name, tensor] of tensorValues.entries()) {
    const shape = tensor.shape;
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
  const payloadView = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  let writeOffset = 0;
  for (const tensor of tensorValues.values()) {
    for (const value of tensor.values) {
      payloadView.setFloat32(writeOffset, value, true);
      writeOffset += 4;
    }
  }
  const out = new Uint8Array(8 + headerBytes.byteLength + payload.byteLength);
  out.set(new Uint8Array(prefix), 0);
  out.set(headerBytes, 8);
  out.set(payload, 8 + headerBytes.byteLength);
  return out;
}

function writeLlama3SourceFixture(fixtureDir) {
  writeFileSync(path.join(fixtureDir, 'config.json'), JSON.stringify({
    architectures: ['TransformerForCausalLM'],
    model_type: 'transformer',
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
      inference: parityInference,
      session: executionSessionDefaults,
      execution: parityExecution,
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

  const gpuProbe = await probeNodeGPU();
  if (!gpuProbe.ready) {
    console.log(`source-runtime-parity.test: skipped (${gpuProbe.reason})`);
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
