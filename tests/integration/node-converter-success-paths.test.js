import assert from 'node:assert/strict';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
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

function readManifest(outputDir) {
  return JSON.parse(readFileSync(path.join(outputDir, 'manifest.json'), 'utf8'));
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
    queryPreAttnScalar: 1,
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
    postAttentionNorm: false,
    preFeedforwardNorm: false,
    postFeedforwardNorm: false,
  },
  ffn: {
    activation: 'gelu',
    gatedActivation: true,
    swigluLimit: null,
  },
  rope: {
    ropeTheta: 10000,
    ropeScalingFactor: 1,
    ropeScalingType: null,
    ropeLocalTheta: null,
    mropeInterleaved: false,
    mropeSection: null,
    partialRotaryFactor: null,
    ropeInterleaved: false,
    yarnBetaFast: null,
    yarnBetaSlow: null,
    yarnOriginalMaxPos: null,
  },
  output: {
    scaleEmbeddings: false,
    tieWordEmbeddings: false,
    embeddingTranspose: false,
    embeddingVocabSize: null,
    finalLogitSoftcapping: null,
    embeddingPostprocessor: null,
  },
  chatTemplate: { type: null, enabled: true },
  layerPattern: { type: 'uniform', globalPattern: null, period: null, offset: null },
};

{
  const fixtureDir = createTempDir('doppler-converter-success-single-');
  const outputDir = path.join(fixtureDir, 'out');
  const reportsDir = createTempDir('doppler-converter-reports-single-');
  const previousReportsDir = process.env.DOPPLER_REPORTS_DIR;
  process.env.DOPPLER_REPORTS_DIR = reportsDir;
  writeGemma2Fixture(fixtureDir, [1]);
  writeFileSync(path.join(fixtureDir, 'tokenizer.json'), JSON.stringify({
    version: '1.0',
    model: {
      vocab: {
        '<unk>': 0,
      },
    },
  }), 'utf8');
  writeFileSync(path.join(fixtureDir, 'tokenizer_config.json'), JSON.stringify({ add_bos_token: true }), 'utf8');
  writeFileSync(path.join(fixtureDir, 'tokenizer.model'), 'tokenizer-model-bytes', 'utf8');
  try {
    const result = await convertSafetensorsDirectory({
      inputDir: fixtureDir,
      converterConfig: {
        output: {
          modelBaseId: 'gemma2-success-single',
          dir: outputDir,
        },
        modelType: 'transformer',
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
    });

    assert.equal(result.outputDir, outputDir);
    assert.ok(typeof result.modelType === 'string');
    assert.ok(result.shardCount >= 1);
    assert.ok(result.tensorCount >= 10);
    assert.equal(result.executionContractArtifact?.schemaVersion, 1);
    assert.equal(result.executionContractArtifact?.ok, true);
    assert.equal(result.executionContractArtifact?.session?.layout, 'contiguous');

    assert.equal(result.layerPatternContractArtifact?.ok, true);
    assert.equal(result.requiredInferenceFieldsArtifact?.ok, true);
    assert.equal(result.report?.suite, 'convert');
    assert.equal(result.report?.executionContractArtifact?.ok, true);

    assert.equal(result.report?.layerPatternContractArtifact?.ok, true);
    assert.equal(result.report?.requiredInferenceFieldsArtifact?.ok, true);
    assert.ok(typeof result.reportInfo?.path === 'string' && result.reportInfo.path.length > 0);

    const reportPath = path.isAbsolute(result.reportInfo.path)
      ? result.reportInfo.path
      : path.resolve(process.cwd(), result.reportInfo.path);
    const reportJson = JSON.parse(readFileSync(reportPath, 'utf8'));
    assert.equal(reportJson.suite, 'convert');
    assert.equal(reportJson.executionContractArtifact?.ok, true);

    assert.equal(reportJson.layerPatternContractArtifact?.ok, true);
    assert.equal(reportJson.requiredInferenceFieldsArtifact?.ok, true);

    const manifest = readManifest(outputDir);
    assert.equal(typeof manifest.modelId, 'string');
    assert.ok(manifest.modelId.startsWith('gemma2-success-single'));
    assert.equal(String(manifest.quantization).toUpperCase(), 'F16');
    assert.equal(manifest.tokenizer?.type, 'bundled');
    assert.equal(manifest.tokenizer?.file, 'tokenizer.json');
    assert.equal(manifest.inference?.schema, 'doppler.execution/v1');
    assert.ok(manifest.inference?.execution?.kernels && typeof manifest.inference.execution.kernels === 'object');
    assert.equal(manifest.tensors?.['lm_head.weight']?.group, 'head');
    assert.equal(manifest.tensors?.['model.norm.weight']?.group, 'head');
    assert.equal(manifest.tensors?.['model.embed_tokens.weight']?.group, 'embed');

    assert.ok(existsSync(path.join(outputDir, 'tokenizer.json')));
    assert.ok(existsSync(path.join(outputDir, 'tokenizer.model')));
  } finally {
    if (previousReportsDir === undefined) {
      delete process.env.DOPPLER_REPORTS_DIR;
    } else {
      process.env.DOPPLER_REPORTS_DIR = previousReportsDir;
    }
    rmSync(fixtureDir, { recursive: true, force: true });
    rmSync(reportsDir, { recursive: true, force: true });
  }
}

{
  const fixtureDir = createTempDir('doppler-converter-success-worker-');
  const outputDir = path.join(fixtureDir, 'out');
  const reportsDir = createTempDir('doppler-converter-reports-worker-');
  const previousReportsDir = process.env.DOPPLER_REPORTS_DIR;
  process.env.DOPPLER_REPORTS_DIR = reportsDir;
  writeGemma2Fixture(fixtureDir, [1, 1]);
  mkdirSync(outputDir, { recursive: true });
  writeFileSync(path.join(outputDir, 'shard_99999.bin'), 'stale-shard', 'utf8');

  const progress = [];
  try {
    const result = await convertSafetensorsDirectory({
      inputDir: fixtureDir,
      converterConfig: {
        output: {
          modelBaseId: 'gemma2-success-worker',
          dir: outputDir,
        },
        modelType: 'transformer',
        quantization: {
          weights: 'f16',
        },
        inference: minimalV1Inference,
        sessionDefaults: minimalV1SessionDefaults,
        execution: minimalV1Execution,
      },
      execution: {
        workers: 2,
        rowChunkRows: 1,
        rowChunkMinTensorBytes: 1,
        maxInFlightJobs: 2,
      },
      onProgress(update) {
        progress.push(update);
      },
    });

    assert.equal(result.outputDir, outputDir);
    assert.ok(typeof result.modelType === 'string');
    assert.ok(result.shardCount >= 1);
    assert.ok(result.tensorCount >= 10);
    assert.equal(result.executionContractArtifact?.schemaVersion, 1);
    assert.equal(result.executionContractArtifact?.ok, true);
    assert.equal(result.executionContractArtifact?.session?.layout, 'contiguous');

    assert.equal(result.layerPatternContractArtifact?.ok, true);
    assert.equal(result.requiredInferenceFieldsArtifact?.ok, true);
    assert.equal(result.report?.suite, 'convert');
    assert.equal(result.report?.executionContractArtifact?.ok, true);

    assert.equal(result.report?.layerPatternContractArtifact?.ok, true);
    assert.equal(result.report?.requiredInferenceFieldsArtifact?.ok, true);
    assert.ok(typeof result.reportInfo?.path === 'string' && result.reportInfo.path.length > 0);

    const reportPath = path.isAbsolute(result.reportInfo.path)
      ? result.reportInfo.path
      : path.resolve(process.cwd(), result.reportInfo.path);
    const reportJson = JSON.parse(readFileSync(reportPath, 'utf8'));
    assert.equal(reportJson.suite, 'convert');
    assert.equal(reportJson.executionContractArtifact?.ok, true);

    assert.equal(reportJson.layerPatternContractArtifact?.ok, true);
    assert.equal(reportJson.requiredInferenceFieldsArtifact?.ok, true);

    const manifest = readManifest(outputDir);
    assert.equal(typeof manifest.modelId, 'string');
    assert.ok(manifest.modelId.startsWith('gemma2-success-worker'));
    assert.equal(String(manifest.quantization).toUpperCase(), 'F16');
    assert.equal(manifest.inference?.schema, 'doppler.execution/v1');

    assert.equal(existsSync(path.join(outputDir, 'shard_99999.bin')), false);
    assert.ok(progress.length > 0);
    assert.ok(
      progress.some((entry) => typeof entry?.message === 'string' && entry.message.includes('requested=')),
      'expected worker summary progress message'
    );
  } finally {
    if (previousReportsDir === undefined) {
      delete process.env.DOPPLER_REPORTS_DIR;
    } else {
      process.env.DOPPLER_REPORTS_DIR = previousReportsDir;
    }
    rmSync(fixtureDir, { recursive: true, force: true });
    rmSync(reportsDir, { recursive: true, force: true });
  }
}

console.log('node-converter-success-paths.test: ok');
