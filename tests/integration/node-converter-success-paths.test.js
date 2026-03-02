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

const castOnlyExecution = {
  steps: [
    {
      id: 'cast.layer.identity',
      op: 'cast',
      phase: 'both',
      section: 'layer',
      src: 'attn_q',
      dst: 'attn_q',
      layers: 'all',
      toDtype: 'f16',
    },
  ],
};

{
  const fixtureDir = createTempDir('doppler-converter-success-single-');
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
        quantization: {
          weights: 'f16',
        },
        inference: {
          execution: castOnlyExecution,
        },
      },
      execution: {
        workers: 1,
      },
    });

    assert.equal(result.outputDir, outputDir);
    assert.equal(result.modelType, 'transformer');
    assert.ok(result.shardCount >= 1);
    assert.ok(result.tensorCount >= 10);

    const manifest = readManifest(outputDir);
    assert.equal(typeof manifest.modelId, 'string');
    assert.ok(manifest.modelId.startsWith('gemma2-success-single'));
    assert.equal(String(manifest.quantization).toUpperCase(), 'F16');
    assert.equal(manifest.tokenizer?.type, 'bundled');
    assert.equal(manifest.tokenizer?.file, 'tokenizer.json');
    assert.equal(manifest.inference?.schema, 'doppler.execution/v0');
    assert.ok(Array.isArray(manifest.inference?.execution?.steps));
    assert.equal(manifest.inference.execution.steps.length, 1);
    assert.equal(manifest.inference.execution.steps[0].op, 'cast');

    assert.ok(existsSync(path.join(outputDir, 'tokenizer.json')));
    assert.ok(existsSync(path.join(outputDir, 'tokenizer.model')));
  } finally {
    rmSync(fixtureDir, { recursive: true, force: true });
  }
}

{
  const fixtureDir = createTempDir('doppler-converter-success-worker-');
  const outputDir = path.join(fixtureDir, 'out');
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
        quantization: {
          weights: 'f16',
        },
        inference: {
          execution: castOnlyExecution,
        },
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
    assert.equal(result.modelType, 'transformer');
    assert.ok(result.shardCount >= 1);
    assert.ok(result.tensorCount >= 10);

    const manifest = readManifest(outputDir);
    assert.equal(typeof manifest.modelId, 'string');
    assert.ok(manifest.modelId.startsWith('gemma2-success-worker'));
    assert.equal(String(manifest.quantization).toUpperCase(), 'F16');
    assert.equal(manifest.inference?.schema, 'doppler.execution/v0');
    assert.equal(manifest.inference.execution.steps.length, 1);
    assert.equal(manifest.inference.execution.steps[0].op, 'cast');

    assert.equal(existsSync(path.join(outputDir, 'shard_99999.bin')), false);
    assert.ok(progress.length > 0);
    assert.ok(
      progress.some((entry) => typeof entry?.message === 'string' && entry.message.includes('requested=')),
      'expected worker summary progress message'
    );
  } finally {
    rmSync(fixtureDir, { recursive: true, force: true });
  }
}

console.log('node-converter-success-paths.test: ok');
