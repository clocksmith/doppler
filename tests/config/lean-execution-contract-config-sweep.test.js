import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { runSweep } from '../../tools/lean-execution-contract-config-sweep.js';

const root = mkdtempSync(path.join(tmpdir(), 'doppler-lean-execution-contract-config-sweep-'));

try {
  const configRoot = path.join(root, 'configs');
  const manifestRoot = path.join(root, 'models');
  const configDir = path.join(configRoot, 'gemma3');
  const embeddingConfigDir = path.join(configRoot, 'embeddinggemma');
  const manifestDir = path.join(manifestRoot, 'local', 'unit-model');
  const mappedManifestDir = path.join(manifestRoot, 'curated', 'mapped-model');
  const embeddingManifestDir = path.join(manifestRoot, 'curated', 'embedding-model');
  mkdirSync(configDir, { recursive: true });
  mkdirSync(embeddingConfigDir, { recursive: true });
  mkdirSync(manifestDir, { recursive: true });
  mkdirSync(mappedManifestDir, { recursive: true });
  mkdirSync(embeddingManifestDir, { recursive: true });

  const zeroDigest = 'sha256:' + '0'.repeat(64);
  const v1Inference = {
    attention: { slidingWindow: null, attnLogitSoftcapping: null, queryKeyNorm: false, attentionOutputGate: false, causal: true, attentionBias: false },
    normalization: { rmsNormWeightOffset: true, rmsNormEps: 1e-6 },
    ffn: { activation: 'gelu', gatedActivation: true, swigluLimit: null },
    rope: { ropeTheta: 1000000, partialRotaryFactor: 1.0, ropeInterleaved: false },
    output: { scaleEmbeddings: true, tieWordEmbeddings: false, embeddingTranspose: false, embeddingVocabSize: null, finalLogitSoftcapping: null },
    chatTemplate: { type: 'gemma' },
    layerPattern: { type: 'every_n', period: 6, offset: 0 },
  };
  const v1SessionDefaults = {
    compute: { defaults: { activationDtype: 'f16', mathDtype: 'f16', accumDtype: 'f32', outputDtype: 'f16' } },
    kvcache: null,
    decodeLoop: null,
  };
  const v1Execution = {
    kernels: { embed: { kernel: 'gather_f16.wgsl', entry: 'main', digest: zeroDigest } },
    preLayer: [['embed', 'embed', 'embed_tokens']],
    decode: [],
    prefill: [],
    postLayer: [],
    policies: { unsupportedPrecision: 'error', dtypeTransition: 'require_cast_step', unresolvedKernel: 'error' },
  };
  writeFileSync(path.join(configDir, 'unit-model.json'), JSON.stringify({
    output: {
      modelBaseId: 'unit-model',
    },
    quantization: {
      weights: 'q4k',
      embeddings: 'f16',
      lmHead: 'f16',
      computePrecision: 'f32',
      q4kLayout: 'row',
    },
    inference: v1Inference,
    sessionDefaults: v1SessionDefaults,
    execution: v1Execution,
  }, null, 2), 'utf8');

  writeFileSync(path.join(manifestDir, 'manifest.json'), JSON.stringify({
    modelId: 'unit-model',
    modelType: 'transformer',
    quantization: 'Q4_K_M',
    quantizationInfo: {
      weights: 'q4k',
    },
    architecture: {
      numLayers: 2,
      hiddenSize: 256,
      intermediateSize: 512,
      numAttentionHeads: 4,
      numKeyValueHeads: 4,
      headDim: 64,
      vocabSize: 1024,
      maxSeqLen: 2048,
      ropeTheta: 1000000,
    },
    inference: {
      presetId: 'gemma3',
      layerPattern: {
        type: 'every_n',
        period: 6,
        offset: 0,
      },
    },
    tensors: {
      'model.embed_tokens.weight': {
        dtype: 'F16',
        shape: [1024, 256],
        role: 'embedding',
      },
      'model.layers.0.self_attn.q_proj.weight': {
        dtype: 'F16',
        shape: [256, 256],
        role: 'matmul',
      },
      'lm_head.weight': {
        dtype: 'F16',
        shape: [1024, 256],
        role: 'head',
      },
    },
  }, null, 2), 'utf8');

  writeFileSync(path.join(configDir, 'mapped-model-f16.json'), JSON.stringify({
    output: {
      modelBaseId: 'mapped-model-f16',
    },
    quantization: {
      weights: 'q4k',
      embeddings: 'f16',
      lmHead: 'f16',
      computePrecision: 'f16',
      q4kLayout: 'row',
    },
    inference: v1Inference,
    sessionDefaults: v1SessionDefaults,
    execution: v1Execution,
  }, null, 2), 'utf8');

  writeFileSync(path.join(mappedManifestDir, 'manifest.json'), JSON.stringify({
    modelId: 'mapped-model',
    modelType: 'transformer',
    quantization: 'Q4_K_M',
    quantizationInfo: {
      weights: 'q4k',
    },
    architecture: {
      numLayers: 2,
      hiddenSize: 256,
      intermediateSize: 512,
      numAttentionHeads: 4,
      numKeyValueHeads: 4,
      headDim: 64,
      vocabSize: 1024,
      maxSeqLen: 2048,
      ropeTheta: 1000000,
    },
    inference: {
      presetId: 'gemma3',
      layerPattern: {
        type: 'every_n',
        period: 6,
        offset: 0,
      },
    },
    tensors: {
      'model.embed_tokens.weight': {
        dtype: 'F16',
        shape: [1024, 256],
        role: 'embedding',
      },
      'model.layers.0.self_attn.q_proj.weight': {
        dtype: 'F16',
        shape: [256, 256],
        role: 'matmul',
      },
      'lm_head.weight': {
        dtype: 'F16',
        shape: [1024, 256],
        role: 'head',
      },
    },
  }, null, 2), 'utf8');

  writeFileSync(path.join(embeddingConfigDir, 'embedding-model.json'), JSON.stringify({
    output: {
      modelBaseId: 'embedding-model',
    },
    quantization: {
      weights: 'q4k',
      embeddings: 'f16',
      lmHead: 'f16',
      computePrecision: 'f32',
      q4kLayout: 'row',
    },
    inference: v1Inference,
    sessionDefaults: v1SessionDefaults,
    execution: v1Execution,
  }, null, 2), 'utf8');

  writeFileSync(path.join(embeddingManifestDir, 'manifest.json'), JSON.stringify({
    modelId: 'embedding-model',
    modelType: 'embedding',
    quantization: 'Q4_K_M',
    quantizationInfo: {
      weights: 'q4k',
    },
    architecture: {
      numLayers: 2,
      hiddenSize: 256,
      intermediateSize: 512,
      numAttentionHeads: 4,
      numKeyValueHeads: 4,
      headDim: 64,
      vocabSize: 1024,
      maxSeqLen: 2048,
      ropeTheta: 1000000,
    },
    inference: {
      presetId: 'embeddinggemma',
      layerPattern: {
        type: 'every_n',
        period: 6,
        offset: 0,
      },
    },
    tensors: {
      'model.embed_tokens.weight': {
        dtype: 'F16',
        shape: [1024, 256],
        role: 'embedding',
      },
      'model.layers.0.self_attn.q_proj.weight': {
        dtype: 'F16',
        shape: [256, 256],
        role: 'matmul',
      },
    },
  }, null, 2), 'utf8');

  writeFileSync(path.join(configDir, 'excluded-template.json'), JSON.stringify({
    output: {
      modelBaseId: 'excluded-template',
    },
    quantization: {
      weights: 'f16',
      embeddings: 'f16',
      lmHead: 'f16',
      computePrecision: 'f16',
    },
  }, null, 2), 'utf8');

  const fixtureMapPath = path.join(configRoot, 'lean-execution-contract-fixtures.json');
  writeFileSync(fixtureMapPath, JSON.stringify({
    schemaVersion: 1,
    source: 'doppler',
    mappings: [
      {
        configPath: path.relative(process.cwd(), path.join(configDir, 'mapped-model-f16.json')),
        manifestPath: path.relative(process.cwd(), path.join(mappedManifestDir, 'manifest.json')),
      },
    ],
    exclusions: [
      {
        configPath: path.relative(process.cwd(), path.join(configDir, 'excluded-template.json')),
        reason: 'fixture excluded on purpose',
      },
    ],
  }, null, 2), 'utf8');

  const summary = await runSweep({
    configRoot,
    manifestRoot,
    fixtureMap: fixtureMapPath,
    json: true,
    check: false,
    requireManifestMatch: true,
  });
  assert.equal(summary.schemaVersion, 1);
  assert.equal(summary.ok, true);
  assert.equal(summary.totals.configs, 4);
  assert.equal(summary.totals.passed, 3);
  assert.equal(summary.totals.explicitSkips, 1);
  assert.equal(summary.results.some((entry) => entry.status === 'pass' && entry.modelId === 'unit-model'), true);
  assert.equal(summary.results.some((entry) => entry.status === 'pass' && entry.modelId === 'mapped-model-f16'), true);
  assert.equal(summary.results.some((entry) => entry.status === 'pass' && entry.modelId === 'embedding-model'), true);
  assert.equal(
    summary.results.some(
      (entry) => entry.status === 'skipped'
        && entry.modelId === 'excluded-template'
        && entry.reason === 'fixture excluded on purpose'
    ),
    true
  );
} finally {
  rmSync(root, { recursive: true, force: true });
}

console.log('lean-execution-contract-config-sweep.test: ok');
