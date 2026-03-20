import assert from 'node:assert/strict';
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
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

function writeGemma2Fixture(fixtureDir, shape = [1]) {
  const elementsPerTensor = shape.reduce((product, value) => product * value, 1);
  const bytesPerTensor = elementsPerTensor * 2;
  const tensorNames = [
    'model.layers.0.self_attn.q_proj.weight',
    'model.layers.0.self_attn.k_proj.weight',
    'model.layers.0.self_attn.v_proj.weight',
    'model.layers.0.self_attn.o_proj.weight',
    'model.layers.0.mlp.gate_proj.weight',
    'model.layers.0.mlp.up_proj.weight',
    'model.layers.0.mlp.down_proj.weight',
    'model.embed_tokens.weight',
    'model.norm.weight',
    'lm_head.weight',
  ];
  const header = {};
  let offset = 0;
  for (const name of tensorNames) {
    header[name] = {
      dtype: 'F16',
      shape,
      data_offsets: [offset, offset + bytesPerTensor],
    };
    offset += bytesPerTensor;
  }
  writeSafetensorsFile(path.join(fixtureDir, 'model.safetensors'), header, offset);
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
}

const ZERO_DIGEST = 'sha256:' + '0'.repeat(64);

const minimalV1Execution = {
  kernels: {
    embed: { kernel: 'gather_f16.wgsl', entry: 'main', digest: ZERO_DIGEST },
    rmsnorm: { kernel: 'rmsnorm.wgsl', entry: 'main', digest: ZERO_DIGEST },
    gemv: { kernel: 'matmul_gemv_subgroup.wgsl', entry: 'main_vec4', digest: ZERO_DIGEST },
    residual: { kernel: 'residual.wgsl', entry: 'main', digest: ZERO_DIGEST },
    gelu: { kernel: 'gelu.wgsl', entry: 'main', digest: ZERO_DIGEST, constants: { HAS_GATE: true } },
    sample: { kernel: 'sample.wgsl', entry: 'sample_single_pass', digest: ZERO_DIGEST },
  },
  preLayer: [['embed', 'embed', 'embed_tokens']],
  decode: [
    ['input_norm', 'rmsnorm'],
    ['q_proj', 'gemv', 'layer.{L}.self_attn.q_proj'],
    ['k_proj', 'gemv', 'layer.{L}.self_attn.k_proj'],
    ['v_proj', 'gemv', 'layer.{L}.self_attn.v_proj'],
    ['o_proj', 'gemv', 'layer.{L}.self_attn.o_proj'],
    ['attn_residual', 'residual'],
    ['gate_proj', 'gemv', 'layer.{L}.mlp.gate_proj'],
    ['up_proj', 'gemv', 'layer.{L}.mlp.up_proj'],
    ['activation', 'gelu'],
    ['down_proj', 'gemv', 'layer.{L}.mlp.down_proj'],
    ['ffn_residual', 'residual'],
  ],
  prefill: [
    ['input_norm', 'rmsnorm'],
    ['q_proj', 'gemv', 'layer.{L}.self_attn.q_proj'],
    ['k_proj', 'gemv', 'layer.{L}.self_attn.k_proj'],
    ['v_proj', 'gemv', 'layer.{L}.self_attn.v_proj'],
    ['o_proj', 'gemv', 'layer.{L}.self_attn.o_proj'],
    ['attn_residual', 'residual'],
    ['gate_proj', 'gemv', 'layer.{L}.mlp.gate_proj'],
    ['up_proj', 'gemv', 'layer.{L}.mlp.up_proj'],
    ['activation', 'gelu'],
    ['down_proj', 'gemv', 'layer.{L}.mlp.down_proj'],
    ['ffn_residual', 'residual'],
  ],
  postLayer: [
    ['final_norm', 'rmsnorm'],
    ['lm_head', 'gemv', 'lm_head'],
    ['sample', 'sample'],
  ],
  policies: {
    unsupportedPrecision: 'error',
    dtypeTransition: 'require_cast_step',
    unresolvedKernel: 'error',
  },
};

const minimalV1SessionDefaults = {
  compute: {
    defaults: {
      activationDtype: 'f16',
      mathDtype: 'f16',
      accumDtype: 'f32',
      outputDtype: 'f16',
    },
  },
  kvcache: null,
  decodeLoop: null,
};

const minimalV1Inference = {
  attention: {
    slidingWindow: null,
    attnLogitSoftcapping: null,
    queryKeyNorm: false,
    attentionOutputGate: false,
    causal: true,
    attentionBias: false,
  },
  normalization: {
    rmsNormWeightOffset: true,
    rmsNormEps: 1e-6,
  },
  ffn: {
    activation: 'gelu',
    gatedActivation: true,
    swigluLimit: null,
  },
  rope: {
    ropeTheta: 10000,
    partialRotaryFactor: 1.0,
    ropeInterleaved: false,
  },
  output: {
    scaleEmbeddings: false,
    tieWordEmbeddings: false,
    embeddingTranspose: false,
    embeddingVocabSize: null,
    finalLogitSoftcapping: null,
  },
  chatTemplate: { type: 'none' },
  layerPattern: { type: 'uniform' },
};

{
  const fixtureDir = createTempDir('doppler-converter-atomicity-');
  const outputDir = path.join(fixtureDir, 'out');
  writeGemma2Fixture(fixtureDir, [1]);
  writeFileSync(path.join(fixtureDir, 'tokenizer.json'), JSON.stringify({
    version: '1.0',
    model: {
      vocab: {
        '<unk>': 0,
      },
    },
  }), 'utf8');
  writeFileSync(path.join(fixtureDir, 'tokenizer.model'), 'tokenizer-model-bytes', 'utf8');
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(path.join(outputDir, 'manifest.json'), JSON.stringify({ modelId: 'stale-manifest' }), 'utf8');
  mkdirSync(path.join(outputDir, 'tokenizer.model'));

  try {
    await assert.rejects(
      () => convertSafetensorsDirectory({
        inputDir: fixtureDir,
        converterConfig: {
          output: {
            modelBaseId: 'gemma2-atomicity',
            dir: outputDir,
          },
          quantization: {
            weights: 'f16',
          },
          inference: minimalV1Inference,
          sessionDefaults: minimalV1SessionDefaults,
          execution: minimalV1Execution,
        },
        execution: {
          workers: 1,
        },
      }),
      /EISDIR|directory/i
    );

    assert.equal(
      existsSync(path.join(outputDir, 'manifest.json')),
      false,
      'failed conversion must not leave a manifest behind'
    );
  } finally {
    rmSync(fixtureDir, { recursive: true, force: true });
  }
}

console.log('node-converter-atomicity.test: ok');
