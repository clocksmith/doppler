import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const root = mkdtempSync(path.join(tmpdir(), 'doppler-lean-execution-contract-config-sweep-'));

try {
  const configRoot = path.join(root, 'configs');
  const manifestRoot = path.join(root, 'models');
  const configDir = path.join(configRoot, 'gemma3');
  const manifestDir = path.join(manifestRoot, 'local', 'unit-model');
  mkdirSync(configDir, { recursive: true });
  mkdirSync(manifestDir, { recursive: true });

  writeFileSync(path.join(configDir, 'unit-model.json'), JSON.stringify({
    output: {
      modelBaseId: 'unit-model',
    },
    presets: {
      model: 'gemma3',
    },
    quantization: {
      weights: 'q4k',
      embeddings: 'f16',
      lmHead: 'f16',
      computePrecision: 'f32',
      q4kLayout: 'row',
    },
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

  const result = spawnSync(
    process.execPath,
    [
      'tools/lean-execution-contract-config-sweep.js',
      '--config-root',
      configRoot,
      '--manifest-root',
      manifestRoot,
      '--json',
      '--no-check',
    ],
    {
      cwd: process.cwd(),
      encoding: 'utf8',
    }
  );

  assert.equal(result.status, 0, result.stderr);
  const summary = JSON.parse(result.stdout);
  assert.equal(summary.schemaVersion, 1);
  assert.equal(summary.ok, true);
  assert.equal(summary.totals.configs, 1);
  assert.equal(summary.totals.passed, 1);
  assert.equal(summary.results[0].status, 'pass');
  assert.equal(summary.results[0].modelId, 'unit-model');
} finally {
  rmSync(root, { recursive: true, force: true });
}

console.log('lean-execution-contract-config-sweep.test: ok');
