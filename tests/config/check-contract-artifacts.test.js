import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { buildContractSummary } from '../../tools/check-contract-artifacts.js';

const summary = await buildContractSummary({
  json: true,
  reportsRoot: '',
  failOnReportContracts: false,
  withLean: false,
  leanCheck: true,
  leanManifestRoot: 'models',
  leanConfigRoot: 'src/config/conversion',
  leanFixtureMap: 'tests/fixtures/lean-execution-contract-fixtures.json',
  leanRequireManifestMatch: false,
});
assert.equal(summary.schemaVersion, 1);
assert.equal(summary.source, 'doppler');
assert.equal(summary.ok, true);
assert.equal(Array.isArray(summary.artifacts), true);
assert.equal(summary.artifacts.some((entry) => entry.id === 'kernelPath' && entry.ok === true), true);
assert.equal(summary.artifacts.some((entry) => entry.id === 'layerPattern' && entry.ok === true), true);

const root = mkdtempSync(path.join(tmpdir(), 'doppler-check-contract-artifacts-'));
try {
  const configRoot = path.join(root, 'configs');
  const manifestRoot = path.join(root, 'models');
  const configDir = path.join(configRoot, 'gemma3');
  const manifestDir = path.join(manifestRoot, 'local', 'unit-model');
  mkdirSync(configDir, { recursive: true });
  mkdirSync(manifestDir, { recursive: true });

  const zeroDigest = 'sha256:' + '0'.repeat(64);
  const minimalV1Config = {
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
    inference: {
      attention: { slidingWindow: null, attnLogitSoftcapping: null, queryKeyNorm: false, attentionOutputGate: false, causal: true, attentionBias: false },
      normalization: { rmsNormWeightOffset: true, rmsNormEps: 1e-6 },
      ffn: { activation: 'gelu', gatedActivation: true, swigluLimit: null },
      rope: { ropeTheta: 1000000, partialRotaryFactor: 1.0, ropeInterleaved: false },
      output: {
        scaleEmbeddings: true,
        tieWordEmbeddings: false,
        embeddingTranspose: false,
        embeddingVocabSize: null,
        embeddingPostprocessor: null,
        finalLogitSoftcapping: null,
      },
      chatTemplate: { type: 'gemma' },
      layerPattern: { type: 'every_n', period: 6, offset: 0 },
    },
    sessionDefaults: {
      compute: { defaults: { activationDtype: 'f16', mathDtype: 'f16', accumDtype: 'f32', outputDtype: 'f16' } },
      kvcache: null,
      decodeLoop: null,
    },
    execution: {
      kernels: { embed: { kernel: 'gather_f16.wgsl', entry: 'main', digest: zeroDigest } },
      preLayer: [['embed', 'embed', 'embed_tokens']],
      decode: [],
      prefill: [],
      postLayer: [],
      policies: { unsupportedPrecision: 'error', dtypeTransition: 'require_cast_step', unresolvedKernel: 'error' },
    },
  };
  writeFileSync(path.join(configDir, 'unit-model.json'), JSON.stringify(minimalV1Config, null, 2), 'utf8');

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

  const fixtureMapPath = path.join(configRoot, 'lean-execution-contract-fixtures.json');
  writeFileSync(fixtureMapPath, JSON.stringify({
    schemaVersion: 1,
    source: 'doppler',
    mappings: [],
    exclusions: [],
  }, null, 2), 'utf8');

  const leanSummary = await buildContractSummary({
    json: true,
    reportsRoot: '',
    failOnReportContracts: false,
    withLean: true,
    leanCheck: false,
    leanManifestRoot: manifestRoot,
    leanConfigRoot: configRoot,
    leanFixtureMap: fixtureMapPath,
    leanRequireManifestMatch: true,
  });
  assert.equal(leanSummary.ok, true);
  assert.equal(leanSummary.lean?.manifestSweep?.ok, true);
  assert.equal(leanSummary.lean?.configSweep?.ok, true);
  assert.equal(
    leanSummary.artifacts.some((entry) => entry.id === 'leanExecutionContractManifests' && entry.ok === true),
    true
  );
  assert.equal(
    leanSummary.artifacts.some((entry) => entry.id === 'leanExecutionContractConfigs' && entry.ok === true),
    true
  );
} finally {
  rmSync(root, { recursive: true, force: true });
}

console.log('check-contract-artifacts.test: ok');
