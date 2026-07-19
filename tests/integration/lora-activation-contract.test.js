import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

const { activateLoRAFromTrainingOutput } = await import('../../src/client/runtime/model-manager.js');
const { activateLoRAFromTrainingOutputForPipeline } = await import('../../src/client/runtime/lora.js');

function createSafetensors(tensors) {
  let offset = 0;
  const header = {};
  const chunks = [];
  for (const tensor of tensors) {
    const values = Float32Array.from(tensor.values);
    const data = Buffer.from(values.buffer);
    header[tensor.name] = {
      dtype: 'F32',
      shape: tensor.shape,
      data_offsets: [offset, offset + data.byteLength],
    };
    offset += data.byteLength;
    chunks.push(data);
  }
  const headerBytes = Buffer.from(JSON.stringify(header), 'utf8');
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(headerBytes.byteLength), 0);
  return Buffer.concat([prefix, headerBytes, ...chunks]);
}

function sha256Hex(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

{
  const result = await activateLoRAFromTrainingOutput({
    adapterManifest: {
      id: 'demo',
      name: 'demo',
      baseModel: 'demo-base',
      rank: 4,
      alpha: 8,
      targetModules: ['q_proj'],
      tensors: [],
    },
  });
  assert.equal(result.activated, false);
  assert.equal(result.reason, 'no_model_loaded');
}

{
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-lora-activation-'));
  const weights = createSafetensors([
    {
      name: 'base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight',
      shape: [1, 1],
      values: [7],
    },
    {
      name: 'base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight',
      shape: [1, 1],
      values: [11],
    },
  ]);
  await fs.writeFile(path.join(dir, 'adapters.safetensors'), weights);
  const manifest = {
    id: 'columbo-peft-demo',
    name: 'columbo-peft-demo',
    baseModel: 'gemma4-e2b-it',
    rank: 1,
    alpha: 2,
    targetModules: ['q_proj'],
    weightsFormat: 'safetensors',
    weightsPath: 'adapters.safetensors',
    checksum: sha256Hex(weights),
    checksumAlgorithm: 'sha256',
  };
  const manifestPath = path.join(dir, 'doppler-adapter-manifest.json');
  await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2));
  const pipeline = {
    manifest: { modelId: 'gemma4-e2b-it' },
    weights: new Map([
      ['layer_0', { qProj: {} }],
    ]),
    lora: null,
    setLoRAAdapter(adapter) {
      this.lora = adapter;
    },
    getActiveLoRA() {
      return this.lora;
    },
  };

  const result = await activateLoRAFromTrainingOutputForPipeline(pipeline, {
    adapterManifestPath: manifestPath,
  });

  assert.equal(result.activated, true);
  assert.equal(result.adapterName, 'columbo-peft-demo');
  assert.equal(result.source, 'adapterManifestPath:file');
  assert.equal(pipeline.lora.layers.get(0).q_proj.scale, 2);

  await assert.rejects(
    () => activateLoRAFromTrainingOutputForPipeline(pipeline, {
      adapterManifest: {
        ...manifest,
        baseModel: 'different-base-model',
        weightsPath: undefined,
        checksum: undefined,
        tensors: [
          {
            name: 'layers.0.q_proj.lora_a',
            shape: [1, 1],
            dtype: 'f32',
            data: [7],
          },
          {
            name: 'layers.0.q_proj.lora_b',
            shape: [1, 1],
            dtype: 'f32',
            data: [11],
          },
        ],
      },
    }),
    /targets base model "different-base-model", but the loaded model is "gemma4-e2b-it"/
  );
  assert.equal(pipeline.lora.name, 'columbo-peft-demo');

  await assert.rejects(
    () => activateLoRAFromTrainingOutputForPipeline(pipeline, {
      adapterManifest: {
        ...manifest,
        targetModules: ['up_proj'],
        weightsPath: undefined,
        checksum: undefined,
        tensors: [
          {
            name: 'layers.0.up_proj.lora_a',
            shape: [1, 1],
            dtype: 'f32',
            data: [7],
          },
          {
            name: 'layers.0.up_proj.lora_b',
            shape: [1, 1],
            dtype: 'f32',
            data: [11],
          },
        ],
      },
    }),
    /targets up_proj at layer 0, but the loaded model has no compatible weight/
  );
  assert.equal(pipeline.lora.name, 'columbo-peft-demo');
}

console.log('lora-activation-contract.test: ok');
