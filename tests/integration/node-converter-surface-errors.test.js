import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { convertSafetensorsDirectory } from '../../src/tooling/node-converter.js';

function createTempDir(prefix) {
  return mkdtempSync(path.join(tmpdir(), prefix));
}

function writeSafetensorsFile(filePath, headerObject, payloadByteLength = 0) {
  const headerJson = JSON.stringify(headerObject);
  const headerBytes = Buffer.from(headerJson, 'utf8');
  const headerPrefix = Buffer.alloc(8);
  headerPrefix.writeBigUInt64LE(BigInt(headerBytes.length), 0);
  const payload = Buffer.alloc(payloadByteLength);
  writeFileSync(filePath, Buffer.concat([headerPrefix, headerBytes, payload]));
}

function writeSingleWeightSafetensors(filePath) {
  writeSafetensorsFile(filePath, {
    'model.embed_tokens.weight': {
      dtype: 'F16',
      shape: [1],
      data_offsets: [0, 2],
    },
  }, 2);
}

function writeMinimalGemma2Safetensors(filePath) {
  writeSafetensorsFile(filePath, {
    'model.layers.0.self_attn.q_proj.weight': { dtype: 'F16', shape: [1], data_offsets: [0, 2] },
    'model.layers.0.self_attn.k_proj.weight': { dtype: 'F16', shape: [1], data_offsets: [2, 4] },
    'model.layers.0.self_attn.v_proj.weight': { dtype: 'F16', shape: [1], data_offsets: [4, 6] },
    'model.layers.0.self_attn.o_proj.weight': { dtype: 'F16', shape: [1], data_offsets: [6, 8] },
    'model.layers.0.mlp.gate_proj.weight': { dtype: 'F16', shape: [1], data_offsets: [8, 10] },
    'model.layers.0.mlp.up_proj.weight': { dtype: 'F16', shape: [1], data_offsets: [10, 12] },
    'model.layers.0.mlp.down_proj.weight': { dtype: 'F16', shape: [1], data_offsets: [12, 14] },
    'model.embed_tokens.weight': { dtype: 'F16', shape: [1], data_offsets: [14, 16] },
    'model.norm.weight': { dtype: 'F16', shape: [1], data_offsets: [16, 18] },
    'lm_head.weight': { dtype: 'F16', shape: [1], data_offsets: [18, 20] },
  }, 20);
}

await assert.rejects(
  () => convertSafetensorsDirectory({}),
  /node convert: inputDir is required\./
);

await assert.rejects(
  () => convertSafetensorsDirectory({
    inputDir: '/tmp/in',
    converterConfig: [],
  }),
  /node convert: converterConfig must be an object when provided\./
);

{
  const fixtureDir = createTempDir('doppler-converter-execution-invalid-');
  try {
    await assert.rejects(
      () => convertSafetensorsDirectory({
        inputDir: fixtureDir,
        execution: 'invalid',
      }),
      /node convert: execution must be an object when provided\./
    );
  } finally {
    rmSync(fixtureDir, { recursive: true, force: true });
  }
}

{
  const fixtureDir = createTempDir('doppler-converter-workers-invalid-');
  try {
    await assert.rejects(
      () => convertSafetensorsDirectory({
        inputDir: fixtureDir,
        execution: {
          workers: 0,
        },
      }),
      /node convert: execution\.workers must be a positive integer\./
    );
  } finally {
    rmSync(fixtureDir, { recursive: true, force: true });
  }
}

{
  const fixtureDir = createTempDir('doppler-converter-file-input-');
  const inputPath = path.join(fixtureDir, 'weights.bin');
  writeFileSync(inputPath, 'not-a-gguf', 'utf8');
  try {
    await assert.rejects(
      () => convertSafetensorsDirectory({
        inputDir: inputPath,
        converterConfig: {
          output: {
            dir: path.join(fixtureDir, 'out'),
          },
        },
      }),
      /node convert: inputDir must be a directory containing safetensors files or a \.gguf file path\./
    );
  } finally {
    rmSync(fixtureDir, { recursive: true, force: true });
  }
}

{
  const fixtureDir = createTempDir('doppler-converter-invalid-config-');
  writeFileSync(path.join(fixtureDir, 'config.json'), '{not-json', 'utf8');
  try {
    await assert.rejects(
      () => convertSafetensorsDirectory({
        inputDir: fixtureDir,
        converterConfig: {
          output: {
            dir: path.join(fixtureDir, 'out'),
          },
        },
      }),
      /Invalid JSON in config\.json \(config\.json\):/
    );
  } finally {
    rmSync(fixtureDir, { recursive: true, force: true });
  }
}

{
  const fixtureDir = createTempDir('doppler-converter-missing-weights-');
  writeFileSync(path.join(fixtureDir, 'config.json'), JSON.stringify({ model_type: 'gemma2' }), 'utf8');
  try {
    await assert.rejects(
      () => convertSafetensorsDirectory({
        inputDir: fixtureDir,
        converterConfig: {
          output: {
            dir: path.join(fixtureDir, 'out'),
          },
        },
      }),
      /ENOENT|no such file/i
    );
  } finally {
    rmSync(fixtureDir, { recursive: true, force: true });
  }
}

{
  const fixtureDir = createTempDir('doppler-converter-invalid-index-');
  writeFileSync(path.join(fixtureDir, 'config.json'), JSON.stringify({ model_type: 'gemma2' }), 'utf8');
  writeFileSync(path.join(fixtureDir, 'model.safetensors.index.json'), '{not-json', 'utf8');
  try {
    await assert.rejects(
      () => convertSafetensorsDirectory({
        inputDir: fixtureDir,
        converterConfig: {
          output: {
            dir: path.join(fixtureDir, 'out'),
          },
        },
      }),
      /Invalid JSON in model\.safetensors\.index\.json/
    );
  } finally {
    rmSync(fixtureDir, { recursive: true, force: true });
  }
}

{
  const fixtureDir = createTempDir('doppler-converter-diffusion-layout-');
  writeFileSync(path.join(fixtureDir, 'model_index.json'), JSON.stringify({}), 'utf8');
  try {
    await assert.rejects(
      () => convertSafetensorsDirectory({
        inputDir: fixtureDir,
        converterConfig: {
          output: {
            dir: path.join(fixtureDir, 'out'),
          },
        },
      }),
      /Unsupported diffusion layout/
    );
  } finally {
    rmSync(fixtureDir, { recursive: true, force: true });
  }
}

{
  const fixtureDir = createTempDir('doppler-converter-diffusion-config-missing-');
  writeFileSync(path.join(fixtureDir, 'model_index.json'), JSON.stringify({
    transformer: ['diffusers', 'Flux2Transformer2DModel'],
    text_encoder: ['transformers', 'Qwen3ForCausalLM'],
    tokenizer: ['transformers', 'Qwen2TokenizerFast'],
    vae: ['diffusers', 'AutoencoderKLFlux2'],
    scheduler: ['diffusers', 'FlowMatchEulerDiscreteScheduler'],
  }), 'utf8');
  try {
    await assert.rejects(
      () => convertSafetensorsDirectory({
        inputDir: fixtureDir,
        converterConfig: {
          output: {
            dir: path.join(fixtureDir, 'out'),
          },
        },
      }),
      /Missing transformer config \(transformer\/config\.json\)/
    );
  } finally {
    rmSync(fixtureDir, { recursive: true, force: true });
  }
}

{
  const fixtureDir = createTempDir('doppler-converter-unknown-family-');
  writeFileSync(path.join(fixtureDir, 'config.json'), JSON.stringify({
    architectures: ['GemmaForCausalLM'],
    model_type: 'gemma',
    num_hidden_layers: 1,
    hidden_size: 1,
    num_attention_heads: 1,
    num_key_value_heads: 1,
    head_dim: 1,
    intermediate_size: 1,
    vocab_size: 10,
    max_position_embeddings: 8,
    bos_token_id: 1,
    eos_token_id: 2,
  }), 'utf8');
  writeSingleWeightSafetensors(path.join(fixtureDir, 'model.safetensors'));
  try {
    await assert.rejects(
      () => convertSafetensorsDirectory({
        inputDir: fixtureDir,
        converterConfig: {
          output: {
            modelBaseId: 'gemma-test',
            dir: path.join(fixtureDir, 'out'),
          },
        },
      }),
      /Unknown model family/
    );
  } finally {
    rmSync(fixtureDir, { recursive: true, force: true });
  }
}

{
  const fixtureDir = createTempDir('doppler-converter-kernel-digest-');
  writeFileSync(path.join(fixtureDir, 'config.json'), JSON.stringify({
    architectures: ['Gemma2ForCausalLM'],
    model_type: 'gemma2',
    num_hidden_layers: 1,
    hidden_size: 1,
    num_attention_heads: 1,
    num_key_value_heads: 1,
    head_dim: 1,
    intermediate_size: 1,
    vocab_size: 10,
    max_position_embeddings: 8,
    bos_token_id: 1,
    eos_token_id: 2,
    rms_norm_eps: 1e-6,
  }), 'utf8');
  writeMinimalGemma2Safetensors(path.join(fixtureDir, 'model.safetensors'));
  try {
    await assert.rejects(
      () => convertSafetensorsDirectory({
        inputDir: fixtureDir,
        converterConfig: {
          output: {
            modelBaseId: 'gemma2-test',
            dir: path.join(fixtureDir, 'out'),
          },
        },
      }),
      /No kernel content digest registered/
    );
  } finally {
    rmSync(fixtureDir, { recursive: true, force: true });
  }
}

{
  const fixtureDir = createTempDir('doppler-converter-worker-policy-invalid-');
  try {
    await assert.rejects(
      () => convertSafetensorsDirectory({
        inputDir: fixtureDir,
        execution: {
          workers: 1,
          workerCountPolicy: 'invalid',
        },
      }),
      /execution\.workerCountPolicy must be "cap" or "error"\./
    );
  } finally {
    rmSync(fixtureDir, { recursive: true, force: true });
  }
}

{
  const fixtureDir = createTempDir('doppler-converter-missing-model-id-');
  writeFileSync(path.join(fixtureDir, 'config.json'), JSON.stringify({
    architectures: ['Gemma2ForCausalLM'],
    model_type: 'gemma2',
    num_hidden_layers: 1,
    hidden_size: 1,
    num_attention_heads: 1,
    num_key_value_heads: 1,
    head_dim: 1,
    intermediate_size: 1,
    vocab_size: 10,
    max_position_embeddings: 8,
    bos_token_id: 1,
    eos_token_id: 2,
    rms_norm_eps: 1e-6,
  }), 'utf8');
  writeMinimalGemma2Safetensors(path.join(fixtureDir, 'model.safetensors'));
  try {
    await assert.rejects(
      () => convertSafetensorsDirectory({
        inputDir: fixtureDir,
        converterConfig: {
          output: {
            dir: path.join(fixtureDir, 'out'),
          },
          inference: {
            sessionDefaults: {
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
            },
            execution: {
              steps: [
                {
                  id: 'cast.identity',
                  op: 'cast',
                  phase: 'both',
                  section: 'layer',
                  src: 'attn_q',
                  dst: 'attn_q',
                  layers: 'all',
                  toDtype: 'f16',
                },
              ],
            },
          },
        },
      }),
      /modelId is required/
    );
  } finally {
    rmSync(fixtureDir, { recursive: true, force: true });
  }
}

{
  const fixtureDir = createTempDir('doppler-converter-missing-output-dir-');
  writeFileSync(path.join(fixtureDir, 'config.json'), JSON.stringify({
    architectures: ['Gemma2ForCausalLM'],
    model_type: 'gemma2',
    num_hidden_layers: 1,
    hidden_size: 1,
    num_attention_heads: 1,
    num_key_value_heads: 1,
    head_dim: 1,
    intermediate_size: 1,
    vocab_size: 10,
    max_position_embeddings: 8,
    bos_token_id: 1,
    eos_token_id: 2,
    rms_norm_eps: 1e-6,
  }), 'utf8');
  writeMinimalGemma2Safetensors(path.join(fixtureDir, 'model.safetensors'));
  try {
    await assert.rejects(
      () => convertSafetensorsDirectory({
        inputDir: fixtureDir,
        converterConfig: {
          output: {
            modelBaseId: 'gemma2-missing-output-dir',
          },
          inference: {
            sessionDefaults: {
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
            },
            execution: {
              steps: [
                {
                  id: 'cast.identity',
                  op: 'cast',
                  phase: 'both',
                  section: 'layer',
                  src: 'attn_q',
                  dst: 'attn_q',
                  layers: 'all',
                  toDtype: 'f16',
                },
              ],
            },
          },
        },
      }),
      /outputDir is required/
    );
  } finally {
    rmSync(fixtureDir, { recursive: true, force: true });
  }
}

{
  const fixtureDir = createTempDir('doppler-converter-invalid-safetensors-header-');
  writeFileSync(path.join(fixtureDir, 'config.json'), JSON.stringify({ model_type: 'gemma2' }), 'utf8');
  writeFileSync(path.join(fixtureDir, 'model.safetensors'), 'short', 'utf8');
  try {
    await assert.rejects(
      () => convertSafetensorsDirectory({
        inputDir: fixtureDir,
        converterConfig: {
          output: {
            dir: path.join(fixtureDir, 'out'),
          },
        },
      }),
      /Invalid safetensors header prefix/
    );
  } finally {
    rmSync(fixtureDir, { recursive: true, force: true });
  }
}

{
  const fixtureDir = createTempDir('doppler-converter-invalid-gguf-header-');
  const ggufPath = path.join(fixtureDir, 'model.gguf');
  writeFileSync(ggufPath, Buffer.from('bad-gguf', 'utf8'));
  try {
    await assert.rejects(
      () => convertSafetensorsDirectory({
        inputDir: ggufPath,
        converterConfig: {
          output: {
            modelBaseId: 'bad-gguf',
            dir: path.join(fixtureDir, 'out'),
          },
        },
      }),
      /GGUF|magic|header|invalid/i
    );
  } finally {
    rmSync(fixtureDir, { recursive: true, force: true });
  }
}

{
  const fixtureDir = createTempDir('doppler-converter-invalid-model-id-');
  writeFileSync(path.join(fixtureDir, 'config.json'), JSON.stringify({
    architectures: ['Gemma2ForCausalLM'],
    model_type: 'gemma2',
    num_hidden_layers: 1,
    hidden_size: 1,
    num_attention_heads: 1,
    num_key_value_heads: 1,
    head_dim: 1,
    intermediate_size: 1,
    vocab_size: 10,
    max_position_embeddings: 8,
    bos_token_id: 1,
    eos_token_id: 2,
    rms_norm_eps: 1e-6,
  }), 'utf8');
  writeMinimalGemma2Safetensors(path.join(fixtureDir, 'model.safetensors'));
  try {
    await assert.rejects(
      () => convertSafetensorsDirectory({
        inputDir: fixtureDir,
        converterConfig: {
          output: {
            modelBaseId: '---',
            dir: path.join(fixtureDir, 'out'),
          },
          inference: {
            sessionDefaults: {
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
            },
            execution: {
              steps: [
                {
                  id: 'cast.identity',
                  op: 'cast',
                  phase: 'both',
                  section: 'layer',
                  src: 'attn_q',
                  dst: 'attn_q',
                  layers: 'all',
                  toDtype: 'f16',
                },
              ],
            },
          },
        },
      }),
      /failed to resolve modelId/
    );
  } finally {
    rmSync(fixtureDir, { recursive: true, force: true });
  }
}

{
  const fixtureDir = createTempDir('doppler-converter-sharded-index-path-');
  writeFileSync(path.join(fixtureDir, 'config.json'), JSON.stringify({
    architectures: ['Gemma2ForCausalLM'],
    model_type: 'gemma2',
    num_hidden_layers: 1,
    hidden_size: 1,
    num_attention_heads: 1,
    num_key_value_heads: 1,
    head_dim: 1,
    intermediate_size: 1,
    vocab_size: 10,
    max_position_embeddings: 8,
    bos_token_id: 1,
    eos_token_id: 2,
    rms_norm_eps: 1e-6,
  }), 'utf8');
  writeFileSync(path.join(fixtureDir, 'model.safetensors.index.json'), JSON.stringify({
    metadata: {},
    weight_map: {
      'model.embed_tokens.weight': 'model-00001-of-00001.safetensors',
    },
  }), 'utf8');
  writeSingleWeightSafetensors(path.join(fixtureDir, 'model-00001-of-00001.safetensors'));
  try {
    const result = await convertSafetensorsDirectory({
      inputDir: fixtureDir,
      execution: {
        workers: 1,
      },
      converterConfig: {
        output: {
          modelBaseId: 'gemma2-sharded-index',
          dir: path.join(fixtureDir, 'out'),
          },
          inference: {
            sessionDefaults: {
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
            },
            execution: {
              steps: [
                {
                id: 'cast.identity',
                op: 'cast',
                phase: 'both',
                section: 'layer',
                src: 'attn_q',
                dst: 'attn_q',
                layers: 'all',
                toDtype: 'f16',
              },
            ],
          },
        },
      },
    });
    assert.equal(result.outputDir, path.join(fixtureDir, 'out'));
    assert.ok(result.tensorCount >= 1);
    assert.ok(result.shardCount >= 1);
  } finally {
    rmSync(fixtureDir, { recursive: true, force: true });
  }
}

{
  const fixtureDir = createTempDir('doppler-converter-gpu-cast-explicit-');
  const originalNodeWebgpuModule = process.env.DOPPLER_NODE_WEBGPU_MODULE;
  const originalNavigator = globalThis.navigator;
  const originalGPUBufferUsage = globalThis.GPUBufferUsage;
  const originalGPUShaderStage = globalThis.GPUShaderStage;
  const originalGPUMapMode = globalThis.GPUMapMode;
  writeFileSync(path.join(fixtureDir, 'config.json'), JSON.stringify({
    architectures: ['Gemma2ForCausalLM'],
    model_type: 'gemma2',
    num_hidden_layers: 1,
    hidden_size: 1,
    num_attention_heads: 1,
    num_key_value_heads: 1,
    head_dim: 1,
    intermediate_size: 1,
    vocab_size: 10,
    max_position_embeddings: 8,
    bos_token_id: 1,
    eos_token_id: 2,
    rms_norm_eps: 1e-6,
  }), 'utf8');
  writeMinimalGemma2Safetensors(path.join(fixtureDir, 'model.safetensors'));
  process.env.DOPPLER_NODE_WEBGPU_MODULE = path.join(fixtureDir, 'missing-webgpu-provider.mjs');
  try {
    delete globalThis.navigator;
    delete globalThis.GPUBufferUsage;
    delete globalThis.GPUShaderStage;
    delete globalThis.GPUMapMode;
    await assert.rejects(
      () => convertSafetensorsDirectory({
        inputDir: fixtureDir,
        execution: {
          useGpuCast: true,
          gpuCastMinTensorBytes: 1,
        },
        converterConfig: {
          output: {
            modelBaseId: 'gemma2-gpu-cast',
            dir: path.join(fixtureDir, 'out'),
          },
          inference: {
            sessionDefaults: {
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
            },
            execution: {
              steps: [
                {
                  id: 'cast.identity',
                  op: 'cast',
                  phase: 'both',
                  section: 'layer',
                  src: 'attn_q',
                  dst: 'attn_q',
                  layers: 'all',
                  toDtype: 'f16',
                },
              ],
            },
          },
        },
      }),
      /execution\.useGpuCast requires a WebGPU-capable Node runtime/
    );
  } finally {
    if (originalNavigator === undefined) {
      delete globalThis.navigator;
    } else {
      globalThis.navigator = originalNavigator;
    }
    if (originalGPUBufferUsage === undefined) {
      delete globalThis.GPUBufferUsage;
    } else {
      globalThis.GPUBufferUsage = originalGPUBufferUsage;
    }
    if (originalGPUShaderStage === undefined) {
      delete globalThis.GPUShaderStage;
    } else {
      globalThis.GPUShaderStage = originalGPUShaderStage;
    }
    if (originalGPUMapMode === undefined) {
      delete globalThis.GPUMapMode;
    } else {
      globalThis.GPUMapMode = originalGPUMapMode;
    }
    if (originalNodeWebgpuModule === undefined) {
      delete process.env.DOPPLER_NODE_WEBGPU_MODULE;
    } else {
      process.env.DOPPLER_NODE_WEBGPU_MODULE = originalNodeWebgpuModule;
    }
    rmSync(fixtureDir, { recursive: true, force: true });
  }
}

console.log('node-converter-surface-errors.test: ok');
